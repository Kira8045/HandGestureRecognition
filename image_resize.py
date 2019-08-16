from PIL import Image

def resize_imaage(image_name):
    img=Image.open(image_name)
    bw=100
    wp=float(bw/img.size[1])
    h=int(wp*img.size[0])

    img = img.resize((bw,h),Image.ANTIALIAS)
    img.save(image_name)

if __name__=="__main__":
    for i in range(100):
        resize_imaage("Dataset/thumb_right/tr_"+str(i)+".png")