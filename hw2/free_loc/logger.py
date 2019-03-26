import visdom
import numpy as np
from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, log_dir,vis_http,port):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        self.vis = visdom.Visdom(server=vis_http,port=port)
        print "Start logging..."

    def scalar_summary(self,tag,value,iteration):
        self.writer.add_scalar(tag,value,iteration)

    def img_summary(self,tag,img,iteration):
        self.writer.add_image(tag,img,iteration)

    def hist_summary(self,tag,data,iteration):
        # hist, bin_edges = np.histogram(data)
        self.writer.add_histogram(tag,data)
        # self.writer.add_histogram_raw(tag, min=np.min(data), max=np.max(data), num=np.size(data), sum=np.sum(data), sum_squares=np.sum(np.multiply(data,data)), bucket_limits=bin_edges, bucket_counts=hist, global_step=iteration)

    def vis_img(self,img, title):
        self.vis.image(img, opts=dict(title=title,caption=title))
