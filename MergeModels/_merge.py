import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from .model_merging_methods.merging_methods import MergingMethod
from .utils.utils import set_random_seed, align_tokenizers_and_embeddings
from .utils.load_config import cache_dir

def get_default_args(models_to_merge, pretrained_model_name, **kwargs):
    parser = argparse.ArgumentParser("Interface for merging LLMs")
    parser.add_argument('--models_to_merge', nargs='+', help='list of models that need to be merged')
    parser.add_argument("--pretrained_model_name", type=str, help="name of the pretrained model")

    parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                        choices=["average_merging", "task_arithmetic", "slerp_merging", "stock_merging", "breadcrumbs_merging", "ties_merging", "widen_merging", "mask_merging"])
    parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
    parser.add_argument("--slerp_t", type=float, default=0.5, help="hyperparameter t for slerp merging")
    parser.add_argument("--dot_threshold", type=float, default=0.9995, help="threshold for considering the two vectors as colinear")
    parser.add_argument("--param_density", type=float, default=0.9, help="density of retained parameters, used for breadcrumbs merging")
    parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="mask rate of the smallest-magnitude parameter values")
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight",
                        choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
    parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                        choices=["average_merging", "task_arithmetic", "slerp_merging", "stock_merging", "breadcrumbs_merging", "ties_merging", "widen_merging"])
    parser.add_argument("--above_average_value_ratio", type=float, default=1.0, help="the ratio above average value")
    parser.add_argument("--score_calibration_value", type=float, default=1.0, help="value for score calibration")

    args = parser.parse_args([])
    args.models_to_merge = models_to_merge
    args.pretrained_model_name = pretrained_model_name

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
            setattr(args, key, value)

    return args

def merge_models(args: argparse.Namespace, finetuned_model_names: list, models_to_merge: list, finetuned_tokenizers: list,
                          finetuned_configs: list, logger: logging.Logger, merging_method: MergingMethod):
    """
    merge models by merging method with merging_method_name and save them
    :param args: ArgumentParser, input argument parser
    :param finetuned_model_names: list, names of finetuned models
    :param models_to_merge: list, individual models that need to be merged
    :param finetuned_tokenizers: list of finetuned tokenizers
    :param finetuned_configs: list of finetuned configs
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :return:
    """
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)
    pretrained_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name)

    # align the tokens of pretrained and finetuned tokenizer
    align_tokenizers_and_embeddings(pretrained_model=pretrained_model, pretrained_tokenizer=pretrained_tokenizer,
                                    pretrained_config=pretrained_config, finetuned_models=models_to_merge,
                                    finetuned_tokenizers=finetuned_tokenizers, finetuned_configs=finetuned_configs, logger=logger)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    merged_model = merging_method.get_merged_model(merged_model=pretrained_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[],
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   slerp_t=args.slerp_t,
                                                   dot_threshold=args.dot_threshold,
                                                   param_density=args.param_density,
                                                   param_value_mask_rate=args.param_value_mask_rate,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=args.weight_mask_rates,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   above_average_value_ratio=args.above_average_value_ratio,
                                                   score_calibration_value=args.score_calibration_value)
    
    return merged_model

def merge(finetuned_model_names, pretrained_model_name, merging_method, **kwargs):

    models_to_merge, finetuned_tokenizers, finetuned_configs = [], [], []
    for finetuned_model_name in finetuned_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, device_map='cpu')
        finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
        finetuned_config = AutoConfig.from_pretrained(finetuned_model_name)
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)
        finetuned_configs.append(finetuned_config)

    args = get_default_args(models_to_merge, pretrained_model_name, **kwargs)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if 'weight_mask_rates' not in kwargs:
        args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    merged = merge_models(args=args, finetuned_model_names=finetuned_model_names, models_to_merge=models_to_merge, finetuned_tokenizers=finetuned_tokenizers,
                        finetuned_configs=finetuned_configs, logger=logger, merging_method=merging_method)
    
    return merged

