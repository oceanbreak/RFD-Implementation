import struct

def candidateRects(height, width, constraint=True):
    """
    Function takes height and width of original image patch and
    calculates set of all possible rectangles can be taken.
    Constraint makes possible to manage possible rectangles number, by constrainting their size (if set to True)
    :return: rect set of kind ((ytopleft, xtopleft, ybottomright, xbottomright))
    """
    rect_param = (height, width)
    rect_set = []
    for i in range(rect_param[0]):
        for j in range(rect_param[1]):
            for k in range(i, rect_param[0]):
                for m in range(j, rect_param[1]):
                    y0, x0 = i, j
                    y1, x1 = k, m
                    if (x1-x0+1) > 4 and (y1-y0+1) > 4 and \
                        (x1-x0+1)%2 == 0 and (y1-y0+1)%2 ==0 and \
                        (x1-x0)*(y1-y0) < 1024 or not constraint:
                        rect_set.append(  (y0, x0, y1, x1)  )

    return tuple(rect_set)

def storeInitThreshold(rectangles,filename, init_threshold = 0.0):
    """
    Createsn little-endian binary file of kind:
    (short: ytopleft, xtopleft, ybottomright, xbottomright, float: threshold)
    """
    with open(filename, 'wb') as binary_file:
        for cur_rect in rectangles:
            for cur_channel in range(8):
                entry = struct.pack('<HHHHHf', *cur_rect, cur_channel, init_threshold)
                binary_file.write(entry)
                binary_file.flush()

def writeThresholdValue(filename, rect_index, channel_index, threshold):
    line_size = struct.calcsize('<HHHHHf')
    offset = (rect_index*8 + channel_index)*line_size + struct.calcsize('<HHHHH')
    with open(filename, 'wb') as binary_file:
        binary_file.seek(offset)
        entry = struct.pack('<f', threshold)
        binary_file.write(entry)
        binary_file.flush()

def readEntry(filename, rect_index, channel_index):
    line_size = struct.calcsize('<HHHHHf')
    offset = (rect_index*8 + channel_index)*line_size
    with open(filename, 'rb') as binary_file:
        binary_file.seek(offset)
        data = binary_file.read(line_size)
        print(struct.unpack('<HHHHHf', data))

if __name__ == '__main__':
    rect_set = candidateRects(64, 64)
    print(len(rect_set))
    filename = "Thresholds_Learned"
    #storeInitThreshold(rect_set, filename)
    for i in range(5):
        for j in range(8):
            readEntry(filename, i, j)
    #writeThresholdValue(filename, 8, 4, 0.783)
    #readEntry(filename, 8, 4)
