import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class VCoderConversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _, _, _, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _, _, _, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _, _, _, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _, _, _, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _, _, _, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode, _, _, _, _ = msg
                    if image is not None:
                        if image_process_mode == "Pad":
                            def expand2square(pil_img, background_color=(122, 116, 104)):
                                width, height = pil_img.size
                                if width == height:
                                    return pil_img
                                elif width > height:
                                    result = Image.new(pil_img.mode, (width, width), background_color)
                                    result.paste(pil_img, (0, (width - height) // 2))
                                    return result
                                else:
                                    result = Image.new(pil_img.mode, (height, height), background_color)
                                    result.paste(pil_img, ((height - width) // 2, 0))
                                    return result
                            image = expand2square(image)
                        elif image_process_mode in ["Default", "Crop"]:
                            pass
                        elif image_process_mode == "Resize":
                            image = image.resize((336, 336))
                        else:
                            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if longest_edge != max(image.size):
                            if H > W:
                                H, W = longest_edge, shortest_edge
                            else:
                                H, W = shortest_edge, longest_edge
                            image = image.resize((W, H))
                        if return_pil:
                            images.append(image)
                        else:
                            buffered = BytesIO()
                            image.save(buffered, format="PNG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                            images.append(img_b64_str)
        return images

    def get_segs(self, return_pil=False):
        segs = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, _, _, seg, seg_process_mode, _, _ = msg
                    if seg is not None:
                        if seg_process_mode == "Pad":
                            def expand2square(pil_img, background_color=(122, 116, 104)):
                                width, height = pil_img.size
                                if width == height:
                                    return pil_img
                                elif width > height:
                                    result = Image.new(pil_img.mode, (width, width), background_color)
                                    result.paste(pil_img, (0, (width - height) // 2))
                                    return result
                                else:
                                    result = Image.new(pil_img.mode, (height, height), background_color)
                                    result.paste(pil_img, ((height - width) // 2, 0))
                                    return result
                            seg = expand2square(seg)
                        elif seg_process_mode in ["Default", "Crop"]:
                            pass
                        elif seg_process_mode == "Resize":
                            seg = seg.resize((336, 336))
                        else:
                            raise ValueError(f"Invalid image_process_mode: {seg_process_mode}")
                        max_hw, min_hw = max(seg.size), min(seg.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = seg.size
                        if longest_edge != max(seg.size):
                            if H > W:
                                H, W = longest_edge, shortest_edge
                            else:
                                H, W = shortest_edge, longest_edge
                            seg = seg.resize((W, H))
                        if return_pil:
                            segs.append(seg)
                        else:
                            buffered = BytesIO()
                            seg.save(buffered, format="PNG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                            segs.append(img_b64_str)
        return segs
    
    def get_depths(self, return_pil=False):
        depths = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, _, _, _, _, depth, depth_process_mode = msg
                    if depth is not None:
                        if depth_process_mode == "Pad":
                            def expand2square(pil_img, background_color=(122, 116, 104)):
                                width, height = pil_img.size
                                if width == height:
                                    return pil_img
                                elif width > height:
                                    result = Image.new(pil_img.mode, (width, width), background_color)
                                    result.paste(pil_img, (0, (width - height) // 2))
                                    return result
                                else:
                                    result = Image.new(pil_img.mode, (height, height), background_color)
                                    result.paste(pil_img, ((height - width) // 2, 0))
                                    return result
                            depth = expand2square(depth)
                        elif depth_process_mode in ["Default", "Crop"]:
                            pass
                        elif depth_process_mode == "Resize":
                            depth = depth.resize((336, 336))
                        else:
                            raise ValueError(f"Invalid image_process_mode: {depth_process_mode}")
                        max_hw, min_hw = max(depth.size), min(depth.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = depth.size
                        if longest_edge != max(depth.size):
                            if H > W:
                                H, W = longest_edge, shortest_edge
                            else:
                                H, W = shortest_edge, longest_edge
                            depth = depth.resize((W, H))
                        if return_pil:
                            depths.append(depth)
                        else:
                            buffered = BytesIO()
                            depth.save(buffered, format="PNG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                            depths.append(img_b64_str)
        return depths

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode, seg, seg_process_mode, depth, depth_process_mode = msg
                    if image is not None:
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                        msg = img_str + msg.replace('<image>', '').strip()

                    if seg is not None:
                        W, H = seg.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        seg = seg.resize((W, H))
                        seg_buffered = BytesIO()
                        seg.save(seg_buffered, format="JPEG")
                        seg_b64_str = base64.b64encode(seg_buffered.getvalue()).decode()
                        seg_str = f'<img src="data:image/png;base64,{seg_b64_str}" alt="user upload seg" />'
                        msg = seg_str + msg.replace('<seg>', '').strip()
                    
                    if depth is not None:
                        W, H = depth.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        depth = depth.resize((W, H))
                        depth_buffered = BytesIO()
                        depth.save(depth_buffered, format="JPEG")
                        depth_b64_str = base64.b64encode(depth_buffered.getvalue()).decode()
                        depth_str = f'<img src="data:image/png;base64,{depth_b64_str}" alt="user upload depth" />'
                        msg = depth_str + msg.replace('<depth>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return VCoderConversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v1 = VCoderConversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1 = VCoderConversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


default_conversation = conv_vicuna_v1
conv_templates = {
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llava_v1": conv_llava_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
