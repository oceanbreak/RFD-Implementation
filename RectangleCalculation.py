

def candidateRects(height, width):
    rect_param = (height, width)

    rect_set = []

    for i in range(rect_param[0]-1):
        for j in range(rect_param[1]-1):
            for k in range(i, rect_param[0]):
                for m in range(j, rect_param[1]):
                    x0, y0 = i, j
                    x1, y1 = k, m
                    if (x1-x0) > 4 and (y1-y0) > 4 and \
                        (x1-x0)%2 == 0 and (y1-y0)%2 ==0 and \
                        (x1-x0)*(y1-y0) < 1024:
                        rect_set.append( ( (x0, y0), (x1, y1) ) )

    return tuple(rect_set)

dataset_cut = 5000
for  i in range(312859 // dataset_cut +1):
    begin = i*dataset_cut
    if (i+1)*dataset_cut < 312859:
        end = (i+1) * dataset_cut
    else:
        end = 312859
    print((begin, end))