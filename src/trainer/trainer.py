import gc
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
import torchvision.transforms as transforms

from tools.tokenizers import *
from tools import TrainingLogger, Evaluator, EarlyStopper
from trainer.build import get_model, get_data_loader, get_tokenizers
from utils import RANK, LOGGER, SCHEDULER_MSG, SCHEDULER_TYPE, colorstr, init_seeds
from utils.func_utils import *
from utils.filesys_utils import *
from utils.training_utils import *




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.scheduler_type = self.config.scheduler_type
        self.world_size = len(self.config.device) if self.is_ddp else 1
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path
        self.max_len = self.config.max_len
        self.metrics = self.config.metrics

        assert self.scheduler_type in SCHEDULER_TYPE, \
            SCHEDULER_MSG + f' but got {colorstr(self.scheduler_type)}'

        # init tokenizer, model, dataset, dataloader, etc.
        self.modes = ['train', 'validation'] if self.is_training_mode else ['train', 'validation', 'test']
        self.preparation = prepare_necessary(config)
        self.tokenizer = get_tokenizers(self.config, self.preparation)
        self.dataloaders = get_data_loader(self.config, self.tokenizer, self.modes, self.preparation, self.is_ddp)
        self.encoder, self.decoder = self._init_model(self.config, self.tokenizer, self.mode)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)
        self.evaluator = Evaluator(self.tokenizer)
        self.stopper, self.stop = EarlyStopper(self.config.patience), False

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args
        
        # init criterion, optimizer, etc.
        self.steps = self.config.steps
        self.enc_lr0, self.enc_lrf = self.config.enc_lr0, self.config.enc_lrf
        self.dec_lr0, self.dec_lrf = self.config.dec_lr0, self.config.dec_lrf
        self.epochs = math.ceil(self.steps / len(self.dataloaders['train'])) if self.is_training_mode else 1
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        if self.is_training_mode:
            self.enc_optimizer = optim.Adam(self.encoder.parameters(), lr=self.config.dec_lr0)
            self.dec_optimizer = optim.Adam(self.decoder.parameters(), lr=self.config.dec_lr0)

            # init scheduler
            self.warmup_steps_n = max(0, self.config.warmup_steps)
            if self.scheduler_type == 'cosine':
                self.enc_lf = one_cycle(1, self.enc_lrf, self.steps)
                self.dec_lf = one_cycle(1, self.dec_lrf, self.steps)
            elif self.scheduler_type == 'linear':
                self.enc_lf = lambda x: (1 - (x - self.warmup_steps_n) / (self.steps - self.warmup_steps_n)) * (1.0 - self.enc_lrf) + self.enc_lrf
                self.dec_lf = lambda x: (1 - (x - self.warmup_steps_n) / (self.steps - self.warmup_steps_n)) * (1.0 - self.dec_lrf) + self.dec_lrf
            
            self.enc_scheduler = optim.lr_scheduler.LambdaLR(self.enc_optimizer, lr_lambda=self.enc_lf)
            self.dec_scheduler = optim.lr_scheduler.LambdaLR(self.dec_optimizer, lr_lambda=self.dec_lf)
            if self.is_rank_zero:
                draw_training_lr_curve(self.config, self.enc_lf, self.config.enc_lr0, self.steps, self.warmup_steps_n, self.is_ddp, self.world_size, 'enc_lr_schedule')
                draw_training_lr_curve(self.config, self.dec_lf, self.config.dec_lr0, self.steps, self.warmup_steps_n, self.is_ddp, self.world_size, 'dec_lr_schedule')


    def _init_model(self, config, tokenizer, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            encoder.load_state_dict(checkpoints['model']['encoder'])
            decoder.load_state_dict(checkpoints['model']['decoder'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return encoder, decoder

        # init models
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        encoder, decoder = get_model(config, tokenizer, self.device)

        # resume model
        if do_resume:
            encoder, decoder = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[self.device])
            torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[self.device])
        
        return encoder, decoder


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
            
            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            # Early Stopping
            if self.is_ddp:  # if DDP training
                broadcast_list = [self.stop if self.is_rank_zero else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if not self.is_rank_zero:
                    self.stop = broadcast_list[0]
            
            if self.stop:
                break  # must break all DDP ranks

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.encoder.train()
        self.decoder.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            logging_header = ['CE Loss', 'enc lr', 'dec lr', f'top-{self.config.topk} acc']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, (img, cap, _) in pbar:
            # Warmup
            self.train_cur_step += 1
            if self.train_cur_step <= self.warmup_steps_n:
                self.enc_optimizer.param_groups[0]['lr'] = lr_warmup(self.train_cur_step, self.warmup_steps_n, self.enc_lr0, self.enc_lf)
                self.dec_optimizer.param_groups[0]['lr'] = lr_warmup(self.train_cur_step, self.warmup_steps_n, self.dec_lr0, self.dec_lf)
            enc_cur_lr = self.enc_optimizer.param_groups[0]['lr']
            dec_cur_lr = self.dec_optimizer.param_groups[0]['lr']
            
            batch_size = img.size(0)
            img, cap = img.to(self.device), cap.to(self.device)
            self.enc_optimizer.zero_grad()
            self.dec_optimizer.zero_grad()
            
            enc_output, hidden = self.encoder(img)
            decoder_all_output, decoder_all_score = [], []
            for j in range(self.max_len):
                trg_word = cap[:, j].unsqueeze(1)
                dec_output, hidden, score = self.decoder(trg_word, hidden, enc_output)
                decoder_all_output.append(dec_output)
                if self.config.using_attention:
                    decoder_all_score.append(score)

            decoder_all_output = torch.cat(decoder_all_output, dim=1)

            loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), cap[:, 1:].reshape(-1))
            if self.config.using_attention:
                decoder_all_score = torch.cat(decoder_all_score, dim=2)
                loss += self.config.regularization_lambda * ((1. - torch.sum(decoder_all_score, dim=2)) ** 2).mean()
            acc = topk_accuracy(decoder_all_output[:, :-1, :], cap[:, 1:], self.config.topk, self.tokenizer.eos_token_id)

            loss.backward()
            self.enc_optimizer.step()
            self.dec_optimizer.step()
            
            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'train_loss': loss.item(), 'enc_lr': enc_cur_lr, 'dec_lr': dec_cur_lr, 'topk_acc': acc},
                )
                loss_log = [loss.item(), enc_cur_lr, dec_cur_lr, acc]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)

        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        def _init_log_data_for_vis():
            data4vis = {'id': [], 'trg': [], 'pred': [], 'score': []}
            return data4vis

        def _append_data_for_vis(**kwargs):
            for k, v in kwargs.items():
                if isinstance(v, list):
                    self.data4vis[k].extend(v)
                else: 
                    self.data4vis[k].append(v)

        with torch.no_grad():
            if self.is_rank_zero:
                if not is_training_now:
                    self.data4vis = _init_log_data_for_vis()

                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                logging_header = ['CE Loss'] + self.config.metrics
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)

                self.encoder.eval()
                self.decoder.eval()

                for i, (img, cap, img_id) in pbar:
                    batch_size = img.size(0)
                    img, cap = img.to(self.device), cap.to(self.device)
                    
                    enc_output, hidden = self.encoder(img)
                    decoder_all_output, decoder_all_score = [], []
                    for j in range(self.max_len):
                        trg_word = cap[:, j].unsqueeze(1)
                        dec_output, hidden, score = self.decoder(trg_word, hidden, enc_output)
                        decoder_all_output.append(dec_output)
                        if self.config.using_attention:
                            decoder_all_score.append(score)

                    decoder_all_output = torch.cat(decoder_all_output, dim=1)

                    loss = self.criterion(decoder_all_output[:, :-1, :].reshape(-1, decoder_all_output.size(-1)), cap[:, 1:].reshape(-1))
                    if self.config.using_attention:
                        decoder_all_score = torch.cat(decoder_all_score, dim=2)
                        loss += self.config.regularization_lambda * ((1. - torch.sum(decoder_all_score, dim=2)) ** 2).mean()
                    acc = topk_accuracy(decoder_all_output[:, :-1, :], cap[:, 1:], self.config.topk, self.tokenizer.eos_token_id)

                    targets4metrics = [self.tokenizer.decode(c.tolist()) for c in cap]
                    predictions = [self.tokenizer.decode([self.tokenizer.bos_token_id] + torch.argmax(pred, dim=1).tolist()) for pred in decoder_all_output]
                    metric_results = self.metric_evaluation(loss, acc, predictions, targets4metrics)
                    metric_results['topk_acc'] = acc

                    self.training_logger.update(
                        phase, 
                        epoch, 
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'validation_loss': loss.item()},
                        **metric_results
                    )

                    # logging
                    loss_log = [loss.item()]
                    msg = tuple([f'{epoch+1}/{self.epochs}'] + loss_log + [metric_results[k] for k in self.metrics])
                    pbar.set_description(('%15s' + '%15.4g' * (len(loss_log) + len(self.metrics))) % msg)

                    ids = random.sample(range(batch_size), min(self.config.prediction_print_n, batch_size))
                    for id in ids:
                        print_samples(targets4metrics[id], predictions[id])

                    if not is_training_now:
                        _append_data_for_vis(
                            **{'trg': targets4metrics,
                               'pred': predictions,
                               'score': decoder_all_score.detach().cpu() if self.config.using_attention else None,
                               'id': img_id.tolist()}
                        )

                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(self.wdir, {'encoder': self.encoder, 'decoder': self.decoder})
                    self.training_logger.save_logs(self.save_dir)

                    high_fitness = self.training_logger.model_manager.best_higher
                    low_fitness = self.training_logger.model_manager.best_lower
                    self.stop = self.stopper(epoch + 1, high=high_fitness, low=low_fitness)

        # for attention visualization
        if not is_training_now and self.config.using_attention:
            self.data4vis['score'] = torch.cat(self.data4vis['score'], dim=0)

    
    def metric_evaluation(self, loss, acc, response_pred, response_gt):
        metric_results = {k: 0 for k in self.metrics}
        for m in self.metrics:
            if m == 'ppl':
                metric_results[m] = self.evaluator.cal_ppl(loss.item())
            elif m == 'bleu2':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=2)
            elif m == 'bleu4':
                metric_results[m] = self.evaluator.cal_bleu_score(response_pred, response_gt, n=4)
            elif m == 'nist2':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=2)
            elif m == 'nist4':
                metric_results[m] = self.evaluator.cal_nist_score(response_pred, response_gt, n=4)
            elif m == 'topk_acc':
                continue
            else:
                LOGGER.warning(f'{colorstr("red", "Invalid key")}: {m}')
        
        return metric_results
    
    
    def vis_attention(self, phase, result_num):
        if result_num > len(self.dataloaders[phase].dataset):
            LOGGER.info(colorstr('red', 'The number of results that you want to see are larger than total test set'))
            sys.exit()
        
        # validation
        self.epoch_validate(phase, 0, False)

        if self.config.using_attention:
            vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs') 
            os.makedirs(vis_save_dir, exist_ok=True)
        else:
            LOGGER.warning(colorstr('yellow', 'Your model does not have attention module..'))

        # show results examples
        trans4attn = transforms.Compose([transforms.Resize((252, 252)),
                                         transforms.ToTensor()])
        random.seed(int(1000*time.time())%(2**32))
        targets, preds = self.data4vis['trg'], self.data4vis['pred']
        targets = [' '.join(self.tokenizer.tokenize(s)[1:-1]) for s in targets]
        ids = random.sample(list(range(len(targets))), result_num)
        img_id = [os.path.join(self.config.img_folder, self.preparation['all_pairs'][j][0]) for j in [self.data4vis['id'][i] for i in ids]]
        
        # preprocessing for visualization
        targets, preds= [targets[i] for i in ids], [preds[i] for i in ids]
        preds = [self.tokenizer.tokenize(s)[1:] for s in preds]
        preds = [tok[:-1] if tok[-1] == self.tokenizer.eos_token else tok for tok in preds]
        preds = [' '.join(tok) for tok in preds]
        if self.config.using_attention:
            pred_l = [len(self.tokenizer.tokenize(tok)) for tok in preds]
            attn_img = [self.data4vis['score'][i, :, :l] for i, l in zip(ids, pred_l)]

        # save result figures
        results_save_path = os.path.join(self.config.save_dir, 'vis_outputs', 'vis_attention')
        save_figures(img_id, targets, preds, results_save_path)
        if self.config.using_attention:
            save_attn_figures(img_id, attn_img, preds, results_save_path, trans4attn, self.config.enc_hidden_dim)