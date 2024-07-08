import os
from PIL import Image
from fpdf import FPDF
import glob
pdf = FPDF()
sdir = "imageFolder/"
def process(uname):
    
    imgpath='./output/'+uname+'/'
    imagelist = glob.glob(imgpath+'*.png')
    w,h = 0,0
    i=1
    for img in imagelist:
        fname = img
        if os.path.exists(fname):
            if i == 1:
                cover = Image.open(fname)
                w,h = cover.size
                pdf = FPDF(unit = "pt", format = [w,h])
            image = fname
            pdf.add_page()
            pdf.image(image,0,0,w,h)
            i=i+1
        else:
            print("File not found:", fname)
        print("processed %d" % i)
    pdf.output(uname+"output.pdf", "F")
    print("done")

