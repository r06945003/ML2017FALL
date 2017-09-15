from PIL import Image
import sys
#import numpy
im = Image.open(sys.argv[1] )
#im = Image.open('westbrook.jpg' )
#one = numpy.array(im)
#print (one[0])
x, y = im.size
for m in range(x):
    for n in range(y):
        RGB = im.getpixel((m,n))
        r = RGB[0] // 2
        g = RGB[1] // 2
        b = RGB[2] // 2
        im.putpixel((m,n), (r, g, b))
im.save("Q2.png")
#im2 = Image.open('Q2.jpg')
#two = numpy.array(im)
#print (two[0])