from history01 import VERSION, saved_solutions, saved_points
import sys, os
import igraph
#import  cairo



_saved_solutions = {'./data/tsp_51_1': (442.91912763515961, [0, 5, 2, 28, 45, 9, 10, 47, 26, 6, 36, 12, 30, 23, 34, 24, 41, 27, 3, 46, 8, 4, 35, 13, 7, 19, 40, 42, 11, 18, 16, 44, 14, 15, 38, 50, 39, 43, 29, 21, 37, 20, 25, 1, 31, 48, 49, 17, 32, 22, 33])}
_saved_points = {'./data/tsp_51_1': [(27.0, 68.0), (30.0, 48.0), (43.0, 67.0), (58.0, 48.0), (58.0, 27.0), (37.0, 69.0), (38.0, 46.0), (46.0, 10.0), (61.0, 33.0), (62.0, 63.0), (63.0, 69.0), (32.0, 22.0), (45.0, 35.0), (59.0, 15.0), (5.0, 6.0), (10.0, 17.0), (21.0, 10.0), (5.0, 64.0), (30.0, 15.0), (39.0, 10.0), (32.0, 39.0), (25.0, 32.0), (25.0, 55.0), (48.0, 28.0), (56.0, 37.0), (30.0, 40.0), (37.0, 52.0), (49.0, 49.0), (52.0, 64.0), (20.0, 26.0), (40.0, 30.0), (21.0, 47.0), (17.0, 63.0), (31.0, 62.0), (52.0, 33.0), (51.0, 21.0), (42.0, 41.0), (31.0, 32.0), (5.0, 25.0), (12.0, 42.0), (36.0, 16.0), (52.0, 41.0), (27.0, 23.0), (17.0, 33.0), (13.0, 13.0), (57.0, 58.0), (62.0, 42.0), (42.0, 57.0), (16.0, 57.0), (8.0, 52.0), (7.0, 38.0)]}

RESULTS_DIR = 'results'
try:
    os.mkdir(RESULTS_DIR)
except:
    pass

def main():
    for path in saved_solutions:
        name = os.path.split(path)[-1]
        png_path = os.path.join(RESULTS_DIR, '%s_%02d.png' % (name, VERSION))
        dist, order = saved_solutions[path]
        points = saved_points[path]
        x, y = zip(*points) 
        sx = max(x) - min(x)
        sy = max(y) - min(y)
    
        disorder = list(enumerate(order))
        disorder.sort(key=lambda x: x[1])
        disorder = [i for i, x in disorder]
    
        nodeCount = len(points)
        layout = points # [map(float, line.split()) for line in fin if line]
   
        #sys.stdin.readline()  # skip header
        #order = map(int, sys.stdin.readline().split())
        edges = zip(order, order[1:] + order[:1])
        print edges
        for i in order:
            print points[i] 
        if sorted(order) != range(nodeCount):
            print "something wrong with solution!"

        g = igraph.Graph(edges=edges, directed=True)
        print type(g)
        style = {}
        if True:
            #style["margin"] = 25
            style["layout"] = layout
            style["vertex_size"] = 20
            #style["bbox"] = (sx, sy)
            #style["vertex_label"] = ['%d:%d' % (disorder[i],i) for i in range(nodeCount)]
            style["vertex_label"] = ['%d' % (disorder[i]) for i in range(nodeCount)]
            style["vertex_label_dist"] = 0
            style["vertex_color"] = "white"
            print 
        #png = fn + ".png"  
        #png = 'test.png'
        print png_path
        #png = None
        igraph.plot(g, png_path, **style)


if __name__ == "__main__":
    main()