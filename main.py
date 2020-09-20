import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy
from ignite.handlers import Checkpoint
from ignite.utils import manual_seed, setup_logger
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler

from peleenet import PeleeNet
from criterions import LabelSmoothKLDivLoss


def get_dataflow(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data, "train"),
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data, "val"),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    return train_loader, val_loader


def initialize(args):
    model = PeleeNet(partial_ratio=args.partial_ratio).to(args.device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = LabelSmoothKLDivLoss(device=args.device)
    lr_scheduler = CosineAnnealingScheduler(optimizer, "lr", start_value=args.lr, end_value=0, cycle_size=args.epochs*args.niter_per_epoch)
    return model, optimizer, criterion, lr_scheduler


def log_metrics(logger, epoch, elapsed, metrics):
    logger.info(
        "\nEpoch {} = elapsed: {} - metrics:\n {}".format(
            epoch, elapsed, "\n".join(["\t{}: {}".format(k, v) for k,v in metrics.items()])
        )
    )


def main(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
    args.device = device

    logger = setup_logger(name="ImageNet-Training")
    output_path = Path(os.path.join(args.savedir))
    if not output_path.exists():
        output_path.mkdir(parents=True)

    train_loader, val_loader = get_dataflow(args)
    args.niter_per_epoch = len(train_loader)
    model, optimizer, criterion, lr_scheduler = initialize(args)
    
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger(name="trainer")
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x})

    metrics = {
        "accuracy1": Accuracy(),
        "accuracy5": TopKCategoricalAccuracy(),
        "loss": Loss(criterion),
    }
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    evaluator.logger = setup_logger(name="evaluator")

    @trainer.on(Events.EPOCH_COMPLETED)
    def eval_model(engine):
        epoch = trainer.state.epoch
        evaluator.run(val_loader)
        state = evaluator.state.metrics
        log_metrics(evaluator.logger, epoch, state.times["COMPLETED"], state.metrics)
    
    common.save_best_model_by_val_score(
        output_path, evaluator, model=model, metric_name="accuracy1", n_saved=3, trainer=trainer, tag="test"
    )
    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", metavar="DIR")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N")
    parser.add_argument("-b", "--batch_size", default=128, type=int, metavar="N")
    parser.add_argument("--epochs", default=240, type=int)
    parser.add_argument("--lr", "--learning_rate", default=5e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--wd", "--weight_decay", default=4e-5, type=float)
    parser.add_argument("--savedir", default="saved_models", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--partial_ratio", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
