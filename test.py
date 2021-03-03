import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    # filename = os.path.join(opt.results_dir, opt.name, 'random_label.txt')
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        # randomnum = np.random.random()
        randomnum = 1
        model.set_input(data,randomnum, 0)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        # fp = open(filename,'a+')
        # fp.write(str(img_path)+"        "+"random_label:"+str(randomnum)+'\n')
        # fp.close()
        visualizer.save_images_withlabel(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()
