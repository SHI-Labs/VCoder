import argparse
import glob
import math
import random
import os
from typing import List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from vcoder_llava.constants import (
    DEFAULT_DEPTH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SEG_TOKEN,
    DEPTH_TOKEN_INDEX,
    IMAGE_TOKEN_INDEX,
    SEG_TOKEN_INDEX,
)
from vcoder_llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_depth_seg_token,
    tokenizer_seg_token,
    tokenizer_image_token,
)
from vcoder_llava.model.builder import load_pretrained_model
from vcoder_llava.utils import disable_torch_init
from vcoder_llava.vcoder_conversation import conv_templates, SeparatorStyle


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class VCoderDataset(Dataset):
    def __init__(
        self,
        prompt: Optional[Union[str, List[str]]],
        tokenizer,
        image_dir,
        image_processor,
        seg_image_dir: Optional[str],
        seg_image_processor,
        depth_image_dir: Optional[str],
        depth_image_processor,
        model_config,
        num_chunks: Optional[int] = None,
        chunk_idx: Optional[int] = None,
        conv_mode: str = "llava_v1",
        ext: str =".jpg",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.seg_image_processor = seg_image_processor
        self.depth_image_processor = depth_image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

        if prompt is None:
            self.prompts = ["What objects can be seen in the image?"]
        elif isinstance(prompt, str):
            self.prompts = [prompt]
        else:
            self.prompts = prompt
        
        chunk_data = num_chunks is not None
        if chunk_data and chunk_idx is None:
            raise ValueError("If `num_chunks` is provided, `chunk_idx` must be provided as well.")

        self.images = glob.glob(os.path.join(image_dir, f"*{ext}"))
        if chunk_data:
            self.images = get_chunk(self.images, num_chunks, chunk_idx)

        if seg_image_dir is not None:
            self.seg_images = glob.glob(os.path.join(seg_image_dir, f"*{ext}"))
            if chunk_data:
                self.seg_images = get_chunk(self.seg_images, num_chunks, chunk_idx)
            if len(self.images) != len(self.seg_images):
                raise ValueError(
                    f"The number of images is {len(self.images)} and segmented images is {len(self.seg_images)}, but"
                    f" the number of images and segmented images must be equal."
                )
        else:
            self.seg_images = None
        
        if depth_image_dir is not None:
            self.depth_images = glob.glob(os.path.join(depth_image_dir, f"*{ext}"))
            if chunk_data:
                self.depth_images = get_chunk(self.depth_images, num_chunks, chunk_idx)
            if len(self.images) != len(self.depth_images):
                raise ValueError(
                    f"The number of images is {len(self.images)} and depth images is {len(self.depth_images)}, but the"
                    f" number of images and depth images must be equal."
                )
        else:
            self.depth_images = None

    def __getitem__(self, index):
        image_file = self.images[index]
        if self.seg_images is not None:
            seg_image_file = self.seg_images[index]
        else:
            seg_image_file = None
        if self.depth_images is not None:
            depth_image_file = self.depth_images[index]
        else:
            depth_image_file = None
        
        prompt = random.choice(self.prompts)
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        image = Image.open(os.path.join(image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        if seg_image_file is not None:
            seg_image = Image.open(os.path.join(seg_image_file)).convert('RGB')
            seg_image_tensor = process_images([seg_image], self.seg_image_processor, self.model_config)[0]
            prompt = DEFAULT_SEG_TOKEN + '\n' + prompt
        else:
            seg_image_tensor = image_tensor
        if depth_image_file is not None:
            depth_image = Image.open(os.path.join(depth_image_file)).convert('RGB')
            depth_image_tensor = process_images([depth_image], self.depth_image_processor, self.model_config)[0]
            prompt = DEFAULT_DEPTH_TOKEN + '\n' + prompt
        else:
            depth_image_tensor = image_tensor
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        conv_prompt = conv.get_prompt()

        if seg_image_file is None and depth_image_file is None:
            input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        elif seg_image_file is not None and depth_image_file is None:
            input_ids = tokenizer_seg_token(
                conv_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, return_tensors='pt'
            )
        else:
            input_ids = tokenizer_depth_seg_token(
                conv_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, SEG_TOKEN_INDEX, DEPTH_TOKEN_INDEX, return_tensors='pt'
            )
        
        return input_ids, image_tensor, seg_image_tensor, depth_image_tensor, image_file.split("/")[-1], prompt

    def __len__(self):
        return len(self.images)


def create_dataloader(
    prompt,
    tokenizer,
    image_dir,
    image_processor,
    seg_image_dir,
    seg_image_processor,
    depth_image_dir,
    depth_image_processor,
    model_config,
    num_chunks: Optional[int] = None,
    chunk_idx: Optional[int] = None,
    conv_mode: str = "llava_v1",
    ext: str =".jpg",
    batch_size: int = 1,
    num_workers: int = 4,
):
    dataset = VCoderDataset(
        prompt,
        tokenizer,
        image_dir,
        image_processor,
        seg_image_dir,
        seg_image_processor,
        depth_image_dir,
        depth_image_processor,
        model_config,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
        conv_mode=conv_mode,
        ext=ext,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader


def postprocess_output(output: str, stop_str: str):
    output = output.strip()
    if output.endswith(stop_str):
        output = output[:-len(stop_str)]
    output = output.strip()
    output = output.strip('\n')
    return output


def evaluate_model(
    model,
    dataloader,
    tokenizer,
    output_file,
    conv_mode: str = "llava_v1",
    temperature: float = 0.2,
    top_p: Optional[float] = None,
    num_beams: int = 1,
    max_new_tokens: int = 512,
):
    model.eval()
    stop_str = (
        conv_templates[conv_mode].sep
        if conv_templates[conv_mode].sep_style != SeparatorStyle.TWO
        else conv_templates[conv_mode].sep2
    )
    do_sample = True if temperature > 0 else False

    nsamples = len(dataloader)
    desc = "Generating answers..."
    for input_ids, images, seg_images, depth_images, image_files, prompts in tqdm(dataloader, total=nsamples, desc=desc):
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        images = images.to(dtype=torch.float16, device='cuda', non_blocking=True)
        seg_images = seg_images.to(dtype=torch.float16, device='cuda', non_blocking=True)
        depth_images = depth_images.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                segs=seg_images,
                depths=depth_images,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        answers = [postprocess_output(output, stop_str) for output in outputs]

        with open(output_file, 'a') as f:
            for i in range(len(answers)):
                f.write(f'Image: {image_files[i]}\n')
                f.write(f'<<PROMPT>>: {prompts[i]}\n')
                f.write(f'<<ANSWER>>: {answers[i]}\n')
                f.write('-------------------------------------------------------\n')


def main(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    (
        tokenizer,
        model,
        image_processor,
        seg_image_processor,
        depth_image_processor,
        context_len,
    ) = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    
    os.makedirs(args.output_folder, exist_ok=True)
    answer_filename = args.output_file_name
    if args.num_chunks is not None and args.chunk_idx is not None:
        answer_filename += f"_{args.num_chunks}_{args.chunk_idx}"
    answer_filename += ".txt"
    answer_path = os.path.join(args.output_folder, answer_filename)

    dataloader = create_dataloader(
        args.prompt,
        tokenizer,
        args.image_folder,
        image_processor,
        args.seg_image_folder,
        seg_image_processor,
        args.depth_image_folder,
        depth_image_processor,
        model.config,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        conv_mode=args.conv_mode,
        ext=args.extension,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    evaluate_model(
        model,
        dataloader,
        tokenizer,
        answer_path,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="shi-labs/vcoder_ds_llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-folder", type=str, default=None, required=True)
    parser.add_argument("--seg-image-folder", type=str, default=None)
    parser.add_argument("--depth-image-folder", type=str, default=None)
    parser.add_argument("--output-folder", type=str, default=None, required=True)
    parser.add_argument("--output-file-name", type=str, default="output")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--chunk-idx", type=int, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    args = parser.parse_args()

    main(args)
