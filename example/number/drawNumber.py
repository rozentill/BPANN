#! /usr/env/bin
#-*-coding:utf8-*-

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import sys
# 随机数字:
def rndNum():
    return chr(random.randint(48, 57))

# 随机颜色1:
def rndColor():
    return (random.randint(0, 0), random.randint(0, 0), random.randint(0, 0))

# 随机颜色2:
def rndColor2():
    return (random.randint(255, 255), random.randint(255, 255), random.randint(255, 255))

if __name__ == "__main__":

    filename = raw_input("Please enter the file name you want to save for this radomly generated image :")

    # 240 x 60:
    width = 28
    height = 28
    image = Image.new('RGB', (width, height), (255, 255, 255))
    # 创建Font对象:
    font = ImageFont.truetype('Arial.ttf', 25)
    # 创建Draw对象:
    draw = ImageDraw.Draw(image)
    # 填充每个像素:
    for x in range(width):
        for y in range(height):
            draw.point((x, y), fill=rndColor())
    # 输出文字:

    draw.text((7, 0), rndNum(), font=font, fill=rndColor2())
    # 模糊:
    # image = image.filter(ImageFilter.BLUR)
    image.save('./numbers/'+filename+'.jpg', 'jpeg')

    print "The image is saved successfully."
