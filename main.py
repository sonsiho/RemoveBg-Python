from helper import bg

alpha_matting = False
af = 240
ab = 10
ae = 10
az = 1000
model_name = "u2netp"

with open('data/HinhChuKy1.png', 'rb') as fh:
    buf = fh.read()

img_remove = bg.remove(buf, model_name, alpha_matting)

# img_remove2 = remove(img_remove)
# print(img_remove2)
# plt.imshow(img_remove)
# plt.show()

img_remove.save('data/Results/HinhChuKy1_rembg.png')
