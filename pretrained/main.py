from config import parse_args
from pretrain_src.generate_update_vector import run_generate_update_vector
from pretrain_src.rvkd_inference_pretrain import run_rvkd_inference_pretrain
from pretrain_src.train_grad_transformer import run_train_grad_transformer
from finetune_src.finetune_aquarat_qwen_loraxs import run_finetune_aquarat_qwen_loraxs


def run_pipeline(args):
    if args.train_option == "pretrain":
        # Step 1: Generate update vectors
        run_generate_update_vector(args)
        # Step 2: Train gradient transformer
        run_train_grad_transformer(args)
        # Step 3: RVKD inference pretrain
        run_rvkd_inference_pretrain(args)
    elif args.train_option == "finetune":
        run_finetune_aquarat_qwen_loraxs(args)
    else:
        raise ValueError(f"Unsupported train_option")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_pipeline(args)

