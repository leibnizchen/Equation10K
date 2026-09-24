# -*- coding: utf-8 -*-
from pathlib import Path
import json,os
from PIL import Image, ImageDraw, ImageFont
from handright import Template, handwrite
from sqlalchemy.sql.functions import random
import random
from tools import BasicTools


class handwrite_generator(object):
    def __init__(self,hand=0):
        self.template_params = {
            "rate": 4,  # 图片缩放比例
            "default_paper_x": 400,  # 默认纸张宽度 px
            "default_paper_y": 300,  # 默认纸张高度 px
            "default_font": BasicTools.get_ttf_file_path()[1][hand],  # 默认字体文件路径
            "default_img_output_path": "outputs",  # 默认图片输出路径
            "default_font_size": 20,  # 默认字体大小
            "default_line_spacing": 30,  # 默认行间距 px
            "default_top_margin": 10,  # 默认顶部留白 px
            "default_bottom_margin": 10,  # 默认底部留白 px
            "default_left_margin": 10,  # 默认左边留白 px
            "default_right_margin": 10,  # 默认右边留白 px
            "default_word_spacing": 1,  # 默认字间距 px
            "default_line_spacing_sigma": 1,  # 默认行间距随机扰动 px
            "default_font_size_sigma": 1,  # 默认字体大小随机扰动 px
            "default_word_spacing_sigma": 1,  # 默认字间距随机扰动 px
            "default_perturb_x_sigma": 1,  # 默认笔画横向偏移随机扰动 px
            "default_perturb_y_sigma": 1,  # 默认笔画纵向偏移随机扰动 px
            "default_perturb_theta_sigma": 0.0,  # 默认笔画旋转偏移随机扰动 rad
            "default_start_chars": "“（[<",  # 特定字符提前换行，防止出现在行尾
            "default_end_chars": "，。 ",  # 防止特定字符因排版算法的自动换行而出现在行首
            "default_background": (255, 255, 255, 255),  # 默认背景颜色 (透明)
            "default_fill": (0, 0, 0, 255),  # 默认字体填充颜色 (黑色)
            "default_region_x": 0,  # 区域起始x坐标
            "default_region_y": 0,  # 区域起始y坐标


        }
        self.template = None  # 模板
        print(BasicTools.get_ttf_file_path()[1][hand])

    def modify_template_params(self, **kwargs):
        for key, value in kwargs.items():
            self.template_params[key] = value
        self.generate_template()

    def generate_template(self):
        rate = self.template_params["rate"]
        paper_width_logic = self.template_params["default_paper_x"]
        paper_height_logic = self.template_params["default_paper_y"]

        # 处理区域参数
        region_x = self.template_params.get("default_region_x", 0)
        region_y = self.template_params.get("default_region_y", 0)
        region_width = self.template_params.get("default_region_width", paper_width_logic)
        region_height = self.template_params.get("default_region_height", paper_height_logic)

        # 检查是否所有区域参数有效
        use_region = all(key in self.template_params for key in
                         ["default_region_x", "default_region_y", "default_region_width", "default_region_height"])

        if use_region:
            # 计算边距
            left_margin_logic = region_x
            top_margin_logic = region_y
            right_margin_logic = paper_width_logic - region_x - region_width
            bottom_margin_logic = paper_height_logic - region_y - region_height

            # 边距合法性检查
            if right_margin_logic < 0 or bottom_margin_logic < 0:
                raise ValueError("区域参数超出纸张范围")

            left_margin = left_margin_logic * rate
            right_margin = right_margin_logic * rate
            top_margin = top_margin_logic * rate
            bottom_margin = bottom_margin_logic * rate
        else:
            # 使用默认边距
            left_margin = self.template_params["default_left_margin"] * rate
            right_margin = self.template_params["default_right_margin"] * rate
            top_margin = self.template_params["default_top_margin"] * rate
            bottom_margin = self.template_params["default_bottom_margin"] * rate

        # 创建模板
        self.template = Template(
            background=Image.new(
                mode="RGBA",
                size=(
                    paper_width_logic * rate,
                    paper_height_logic * rate
                ),
                color=self.template_params["default_background"]
            ),
            font=ImageFont.truetype(
                self.template_params["default_font"],
                size=self.template_params["default_font_size"] * rate
            ),
            line_spacing=self.template_params["default_line_spacing"] * rate,
            fill=self.template_params["default_fill"],
            left_margin=left_margin,
            top_margin=top_margin,
            right_margin=right_margin,
            bottom_margin=bottom_margin,
            word_spacing=self.template_params["default_word_spacing"] * rate,
            line_spacing_sigma=self.template_params["default_line_spacing_sigma"] * rate,
            font_size_sigma=self.template_params["default_font_size_sigma"] * rate,
            word_spacing_sigma=self.template_params["default_word_spacing_sigma"] * rate,
            start_chars=self.template_params["default_start_chars"],
            end_chars=self.template_params["default_end_chars"],
            perturb_x_sigma=self.template_params["default_perturb_x_sigma"],
            perturb_y_sigma=self.template_params["default_perturb_y_sigma"],
            perturb_theta_sigma=self.template_params["default_perturb_theta_sigma"]
        )

    def generate_image(self, question,text,png_path):
        temp_file_path_dict = {}
        if self.template is None:
            self.generate_template()
        images = handwrite(text, self.template, "outputs")

        for i, im in enumerate(images):

            assert isinstance(im, Image.Image)
            save_path = Path().joinpath(f"{png_path}")
            temp_file_path_dict[i] = save_path

            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype("arial.ttf", 46)
            except IOError:
                font = ImageFont.load_default()  # 默认字体不支持中文

            draw.text((50, 50), f"solve: {question}", fill=(0, 0, 0), font=font)
            im.save(save_path)

        return temp_file_path_dict


if __name__ == '__main__':


   for i in range(0,12):
        print(i)
        datasst = "48÷4+3^2×7"
        generator = handwrite_generator(hand=i)
        data = ["12+3^2×7","12+9×7","12+16","28"]
        generator.modify_template_params(
            default_region_x=50,
            default_region_y=50,
            default_region_width=250,
            default_region_height=250,
            rate=2
        )
        text =   "\n".join( f"={str(step).replace("**","^").replace("*","X").replace("×","X").replace('÷','/').lstrip('=')}" for step in data)
        text= text.replace(" ","")
        generator.generate_image(datasst,text,f"{"handwriting_samples"}/{i}.png")


