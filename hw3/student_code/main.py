import argparse
from student_code.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from student_code.coattention_experiment_runner import CoattentionNetExperimentRunner
import torch

if __name__ == "__main__":
    torch.manual_seed(7)
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str)
    parser.add_argument('--train_question_path', type=str)
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_data_loader_workers', type=int, default=6)
    parser.add_argument('--preprocessing', type=bool, default=False)
    args = parser.parse_args()

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                preprocessing=args.preprocessing)
    experiment_runner.train()


# python -m student_code.main --train_image_dir ~/Downloads/16824_data/train2014 --test_image_dir ~/Downloads/16824_data/val2014 --train_question_path ~/Downloads/16824_data/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json --train_annotation_path ~/Downloads/16824_data/mscoco_train2014_annotations.json --test_annotation_path ~/Downloads/16824_data/mscoco_val2014_annotations.json --test_question_path ~/Downloads/16824_data/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json 