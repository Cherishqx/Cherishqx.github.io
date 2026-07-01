# 在StarVLA框架中加入GenLIP的经验（还有bug没解决）

**！！！误人子弟，谨慎阅读**


1. 首先下载好所需模型，把GenLIP、StarVLA官方代码克隆下来。
2. 在starVLA/starVLA/model/modules/vlm目录下加入GenLIP\.py

   ```Python
   from typing import Optional

   import sys
   sys.path.insert(0, "你放GenLIP代码的路径")

   from GenLIP.veomni.models.transformers.genlip.genlip_modeling import GenLIPModel
   from GenLIP.veomni.models.transformers.genlip.image_processor import GenLIPNaViTImageProcessor
   try:
       from GenLIP.veomni.utils.flex_attn_utils import create_fast_flex_mask_padding
       _FLEX_ATTN_AVAILABLE = True
   except ImportError:
       _FLEX_ATTN_AVAILABLE = False
       print("Warning: create_fast_flex_mask_padding not found, flex attention disabled.")

   import torch
   from starVLA.training.trainer_utils import initialize_overwatch
   from transformers.modeling_outputs import CausalLMOutputWithPast
   from qwen_vl_utils import process_vision_info
   from transformers import AutoConfig, AutoTokenizer
   logger = initialize_overwatch(__name__)

   import torch.nn as nn

   IGNORE_INDEX = -100
   IMAGE_TOKEN_INDEX = 151655
   VIDEO_TOKEN_INDEX = 151656
   DEFAULT_IMAGE_TOKEN = "<image>"
   DEFAULT_VIDEO_TOKEN = "<video>"
   IMAGE_INPUT_INDEX = -200
   _ACTION_TOKEN_MIN = 151936  # how can we know this range?
   _ACTION_TOKEN_MAX = (
       153983  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
   )

   class _GenLIP_Interface(nn.Module):

       def __init__(self, config: Optional[dict] = None, **kwargs):
           """
           Initialize the VLM wrapper.
           Following https://huggingface.co/YanFang/GenLIP-L16-NaViT
           """
           super().__init__()

           qwenvl_config = config.framework.get("qwenvl", {})
           print(f"qwenvl_config:{qwenvl_config}")

           model_id = qwenvl_config.get("base_vlm", "YanFang/GenLIP-L16-NaViT")

           model = GenLIPModel.from_pretrained(model_id,torch_dtype="bfloat16")
           processor = GenLIPNaViTImageProcessor.from_pretrained(model_id)
           tokenizer = AutoTokenizer.from_pretrained(model_id)

           self.model = model
           self.processor = processor
           self.tokenizer = tokenizer
           self.config = config
           self.use_flex_attn = True

       def forward(
           self,
           **kwargs,
       ) -> CausalLMOutputWithPast:
           with torch.autocast("cuda", dtype=torch.bfloat16):
               outputs = self.model(
                   **kwargs,
               )
           # print(outputs.hidden_states[-1].shape)
           return outputs

       def build_qwenvl_inputs(self, images, instructions,solutions=None, **kwargs):
           messages = []
           assert len(images) == len(instructions), "Images and instructions must have the same length"
           for imgs, instruction in zip(images, instructions):
               content = [{"type": "image", "image": img} for img in imgs]

               if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                   CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                   prompt = CoT_prompt.replace("{instruction}", instruction)
               else:
                   prompt = instruction

               content.append({"type": "text", "text": prompt})
               msg = [{"role": "user", "content": content}]

               if solutions is not None:
                   solution = solutions[len(messages)]
                   msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
               messages.append(msg)

           # Prepare text prompts using processor
           # default process is json --> message --> texts --> input_ids
           texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
           all_images = []
           for imgs in images:
               all_images.extend(imgs)

           image_outputs = self.processor(images=all_images if len(all_images) > 0 else None,return_tensors="pt")

           if len(all_images) > 0:
               merge_length = self.processor.merge_size ** 2  # merge_length:每个vision token包含的patch数
               image_grid_thw = image_outputs["image_grid_thw"] #[1,14,14]
               img_idx = 0
               for i in range(len(texts)):
                   while "<|image_pad|>" in texts[i]:
                       num_tokens = int(image_grid_thw[img_idx].prod()) // merge_length
                       texts[i] = texts[i].replace("<|image_pad|>", "<|placeholder|>" * num_tokens, 1)
                       img_idx += 1
                   texts[i] = texts[i].replace("<|placeholder|>", "<|image_pad|>")
               assert img_idx == len(all_images), (
                   f"chat template produced {img_idx} <|image_pad|> tokens for {len(all_images)} images; "
                   "check that the tokenizer's chat_template renders image content"
               )

           text_outputs = self.tokenizer(texts,padding=True,return_tensors="pt")
           image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")  # 按实际特殊token修改
           input_id = text_outputs["input_ids"]
           image_mask = (input_id == image_token_id)  # [B, seq_len], bool

           flex_attn_args = None
           if self.use_flex_attn and _FLEX_ATTN_AVAILABLE:
               B, seq_len = input_id.shape
               device = self.model.device

               flex_indicators = image_mask.long().to(device)   # [B, seq_len]

               sample_ids = (
                   torch.arange(B, device=device)
                   .unsqueeze(1)
                   .expand(B, seq_len)
                   .contiguous()
               )                                                 # [B, seq_len]

               max_seqlen = seq_len
               flex_mask = create_fast_flex_mask_padding(
                   sample_ids,
                   flex_indicators,
                   max_seqlen,
               )
               flex_attn_args = dict(flex_mask=flex_mask, max_seqlen=max_seqlen)

           self.model.image_token_id = image_token_id
           self.model.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
           position_ids, rope_deltas = self.model.get_rope_index(
               input_ids=input_id,
               image_grid_thw=image_outputs["image_grid_thw"],
               attention_mask=text_outputs["attention_mask"],
           )

           input_id[image_mask] = image_token_id

           if len(all_images) > 0:
               expected_vision_tokens = int((image_grid_thw.prod(dim=-1) // merge_length).sum())
               assert int(image_mask.sum()) == expected_vision_tokens, (
                   f"image_mask.sum()={int(image_mask.sum())} != merged vision patches "
                   f"{expected_vision_tokens}; vision features would be mis-aligned"
               )

           batch_input = {
               "input_ids": input_id,
               "attention_mask": text_outputs["attention_mask"],
               "pixel_values": image_outputs["pixel_values"],
               "image_grid_thw": image_outputs["image_grid_thw"],
               "image_mask": image_mask,
               "position_ids":position_ids,
               "rope_deltas":rope_deltas,
               "flex_attn_args": flex_attn_args,
           }
           # if solutions, mask out the non solution tokens in labels --> @JinhuiYE can we mask out system prompt?
           if solutions is not None:
               action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
               action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
               labels = batch_input["input_ids"].clone()
               # For each sequence in the batch, find the first occurrence of an action token.
               for i in range(labels.size(0)):
                   seq = labels[i]
                   # Create a mask for tokens within the action token range.
                   mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                   nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                   if nonzero_indices.numel() > 0:
                       first_action_index = nonzero_indices[0].item()
                       # Mask out all tokens before the first action token.
                       seq[:first_action_index] = IGNORE_INDEX
                   else:
                       # If no action token is found, mask the entire sequence.
                       seq[:] = IGNORE_INDEX
                       RuntimeWarning(
                           "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                       ) 

               labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
               batch_input["labels"] = labels

           return {
               k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
               for k, v in batch_input.items()
           }

   if __name__ == "__main__":
       import argparse
       import os

       from omegaconf import OmegaConf

       parser = argparse.ArgumentParser()
       parser.add_argument(
           "--config_yaml",
           type=str,
           default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
           help="Path to YAML config",
       )
       args, clipargs = parser.parse_known_args()

       if os.getenv("DEBUGPY_ENABLE", "0") == "1":
           import debugpy
           debugpy.listen(("0.0.0.0", 10092))
           print("Rank 0 waiting for debugger attach on port 10092...")
           debugpy.wait_for_client()

       cfg = OmegaConf.load(args.config_yaml)

       model_id = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
       cfg.framework.qwenvl.base_vlm = model_id

       model = _GenLIP_Interface(config=cfg)
       pass
   ```
3. 修改starVLA/starVLA/model/modules/vlm/init\.py文件

   ```Python
   def get_vlm_model(config):
       #...(省略）...
       #加入以下代码
       elif "genlip" in vlm_name.lower():
           from .GenLIP import _GenLIP_Interface

           return _GenLIP_Interface(config)
       #...(省略）...
   ```
4. genlip返回的是最后一层的隐藏状态，starVLA里面的qwen\_vl\_interface的输出都是所有层的隐藏状态

   ```Python
   with torch.autocast("cuda", dtype=torch.bfloat16):
               qwenvl_outputs = self.qwen_vl_interface(
                   **qwen_inputs,
                   output_attentions=False,
                   output_hidden_states=True,
                   return_dict=True,
               )
               # last_hidden_state: [B, seq_len, H]
               last_hidden = qwenvl_outputs.hidden_states[-1]
   ```

   所以要在GenLIP/veomni/models/transformers/genlip/genlip\_modeling\.py里面修改一下相关代码：

   ```Python
   class GenLIPEncoder(nn.Module):
       # ......
       def forward(
           self,
           ...
           output_hidden_states: bool = False,   # ← 新增参数
       ) -> Tuple[torch.Tensor, Optional[List]]:
            ...
               if use_cache:
                   present_key_values.append(present_kv)

               if output_hidden_states:
                   all_hidden_states.append(hidden_states)

           final_hidden = (tuple(all_hidden_states) if output_hidden_states else hidden_states)

           return final_hidden, (present_key_values if use_cache else None)
   class GenLIPModel(GenLIPPreTrainedModel):
       def forward(...):
           ......
               hidden_states, present_key_values = self.visual(
                   input_embeds,
                   cu_seqlens=None,
                   attention_mask=attention_mask,
                   flex_attn_args=flex_attn_args,
                   position_embeddings=position_embeddings,
                   past_key_values=past_key_values,
                   use_cache=use_cache or False,
                   output_hidden_states=True, # 让每一层都输出
               )
               # hidden_states = self.proj(self.ln_post(hidden_states))
               # 新代码
               if isinstance(hidden_states, tuple):
                   all_hidden_states = hidden_states          # 保存所有层
                   hidden_states = hidden_states[-1]          # 取最后一层做后续计算
               else:
                   all_hidden_states = (hidden_states,)

               ......

               return GenLIPModelOutput(
                   loss=loss,
                   logits=logits,
                   hidden_states=all_hidden_states,
                   rope_deltas=rope_deltas,
                   past_key_values=present_key_values,
               )
   ```
5. \(没有及时记录，其它暂时不记得了）
6. 可以运行starVLA/starVLA/model/framework/VLM4A/QwenGR00T\.py来进行调试。

   ```Shell
   python starVLA/model/framework/VLM4A/QwenGR00T.py
   ```

---

主要就是这些步骤，记录一下遇到的一些问题。因为GenLIP\.py是直接对照着QWen2\_5\.py写的，AI辅助，然后再一点点调的bug。

**模型加载：**

```Python
model = GenLIPModel.from_pretrained(model_id,torch_dtype="bfloat16")
processor = GenLIPNaViTImageProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

这里的model和processor还没有集成到transformers里面，都用的是GenLIP代码提供的，tokenizer就用的AutoTokenizer。

**图像占位符：**

```Python
if len(all_images) > 0:
    merge_length = self.processor.merge_size ** 2  # merge_length:每个vision token包含的patch数
    image_grid_thw = image_outputs["image_grid_thw"] #[1,14,14]
    img_idx = 0
    for i in range(len(texts)):
        while "<|image_pad|>" in texts[i]:
            num_tokens = int(image_grid_thw[img_idx].prod()) // merge_length
            texts[i] = texts[i].replace("<|image_pad|>", "<|placeholder|>" * num_tokens, 1)
            img_idx += 1
        texts[i] = texts[i].replace("<|placeholder|>", "<|image_pad|>")
```

**merge\_length：**计算每个 vision token 对应的 patch 数，如果merge\_length =2\*2=4，说明4个patch对应1个token

**\|image\_pad\|：**图像占位符的个数等于应该有的token的个数。

**模型输入：**

```Python
batch_input = {
    "input_ids": input_id,  #文本 token ID 序列，其中图片位置已被展开为对应数量的<|image_pad|>token
    "attention_mask": text_outputs["attention_mask"], #注意力掩码，这里不用输入
    "pixel_values": image_outputs["pixel_values"], #原始图片像素张量，形状通常为(num_images, C, H, W)，输入到ViT视觉编码器
    "image_grid_thw": image_outputs["image_grid_thw"], #每张图片的patch网格尺寸(T, H, W)，用于计算视觉token数量和位置编码
    "image_mask": image_mask, #图片token掩码，用于标记哪些是图像token
    "position_ids":position_ids, #位置编码 ID
    "rope_deltas":rope_deltas, #RoPE位置偏移量
    "flex_attn_args": flex_attn_args, #灵活注意力配置（建议去掉这个参数和对应代码，我加了之后程序跑不起来了）
}
```

\!一定要确保输入是正确的，我之前看位置编码会自动生成就没有输入，结果发现模型里是随便生成的，导致成功率为0

目前还有Bug，但是不知道怎么改了。主要问题是模型会不完全听从指令，明明指令是任务A，执行的却是任务B。