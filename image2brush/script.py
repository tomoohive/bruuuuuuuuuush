from PIL import Image, ImageDraw, ImageFilter

im_rgba = Image.open('./result/brush_crop10.png')
im_back = Image.new('RGBA', im_rgba.size, (0, 0, 0, 0))
im_a = Image.open('./result/mask_crop10.png').convert('L')
im_a_blur = im_a.filter(ImageFilter.GaussianBlur(2))

im = Image.composite(im_rgba, im_back, im_a_blur)
im.save('./result/blur_brush.png')

