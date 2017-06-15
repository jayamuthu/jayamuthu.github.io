.. title: VanangamudiMNIST
.. slug: vanangamudimnist
.. date: 2017-04-27 23:00:00 UTC-03:00
.. tags: deep learning, intro, mnist
.. description:
.. category: neural networks
.. section: neural networks

.. code:: python3

    import torch
    from torch.autograd import Variable

DATASET
-------

.. code:: python3

    dataset = [] #list of tuples (image, label)
    
    zer = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1],
                       ])
    
    one = torch.Tensor([[0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 1],
                       ])
    
    two = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1],
                       ])
    
    thr = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1],
                       ])
    
    fou = torch.Tensor([[0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                       ])
    
    fiv = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1],
                       ])
    
    six = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1],
                       ])
    
    sev = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1],
                       ])
    
    eig = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1],
                       ])
    
    nin = torch.Tensor([[0, 0, 1, 1, 1],
                        [0, 0, 1, 0, 1],
                        [0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1],
                       ])
    
    dataset.append((zer, torch.Tensor([0])))
    dataset.append((one, torch.Tensor([1])))
    dataset.append((two, torch.Tensor([2])))
    dataset.append((thr, torch.Tensor([3])))
    dataset.append((fou, torch.Tensor([4])))
    dataset.append((fiv, torch.Tensor([5])))
    dataset.append((six, torch.Tensor([6])))
    dataset.append((sev, torch.Tensor([7])))
    dataset.append((eig, torch.Tensor([8])))
    dataset.append((nin, torch.Tensor([9])))
    


Take a look into how the data looks like
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    %matplotlib inline
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    from PIL import Image
    
    def showImage(path):
        image = Image.open(path)
        image.show()
    
    fig = plt.figure(1,(10., 50.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2 , 5),
                     axes_pad=0.1)
    
    for i, (data, target) in enumerate(dataset):
        grid[i].matshow(Image.fromarray(data.numpy()))
    plt.show()



.. image::  /images/vanangamudimnist/output_4_0.png


MODEL
-----

.. code:: python3

    from torch import nn
    import torch.nn.functional as F
    import torch.optim as optim
    
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.output_layer = nn.Linear(5*5, 10, bias=False)
    
        def forward(self, x):
            x = self.output_layer(x)
            return F.log_softmax(x)
        

.. code:: python3

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

Training
--------

Train for a single epoch
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    def train(model, optim, dataset):
        model.train()
        avg_loss = 0
        for i, (data, target) in enumerate(dataset):
            data = data.view(1, -1)
            data, target = Variable(data), Variable(target.long())
            optimizer.zero_grad()
            output = model(data)
    
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            
        return avg_loss/len(dataset)

DATASET - MODEL - OUTPUT
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    fig = plt.figure(1, (16., 16.))
    grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 3),
                         axes_pad=0.1)
    
    
    data = [data.view(-1) for data, target in dataset]
    data = torch.stack(data)
    
    target = [target.view(-1) for data, target in dataset]
    target = torch.stack(target).squeeze()
    grid[0].matshow(Image.fromarray(data.numpy()))
    grid[0].set_xlabel('DATASET', fontsize=24)
    
    grid[1].matshow(Image.fromarray(model.output_layer.weight.data.numpy()))
    grid[1].set_xlabel('MODEL', fontsize=24)
    
    output = model(Variable(data))
    grid[2].matshow(Image.fromarray(output.data.numpy()))
    grid[2].set_xlabel('OUTPUT', fontsize=24)
    
    
    plt.show()



.. image::  /images/vanangamudimnist/output_12_0.png


.. code:: python3

    fig = plt.figure(1,(12., 12.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2 , 5),
                     axes_pad=0.1)
    
    for i, (d, t) in enumerate(dataset):
        grid[i].matshow(Image.fromarray(d.numpy()))
        
    plt.show()
    
    fig = plt.figure(1, (100., 10.))
    grid = ImageGrid(fig, 111,
                         nrows_ncols=(len(dataset), 1),
                         axes_pad=0.1)
    
    
    data = [data.view(1, -1) for data, target in dataset]
    
    for i, d in enumerate(data):
        grid[i].matshow(Image.fromarray(d.numpy()))
        grid[i].set_ylabel('{}'.format(i), fontsize=36)




.. image::  /images/vanangamudimnist/output_13_0.png



.. image::  /images/vanangamudimnist/output_13_1.png


How many correct predictions without any training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    pred = output.data.max(1)[1].squeeze()
    correct = pred.eq(target.long()).sum()
    print('correct: {}/{}'.format(correct, len(dataset)))


.. parsed-literal::

    correct: 0/10


lets combine the above two blocks and make a function out of it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    def test_and_print(model, dataset, plot=True):
          
        data = [data.view(-1) for data, target in dataset]
        data = torch.stack(data).squeeze()
    
        target = [target.view(-1) for data, target in dataset]
        target = torch.stack(target).squeeze()
        output = model(Variable(data))
            
        loss = F.nll_loss(output, Variable(target.long()))
        
        dataset_img = Image.fromarray(data.numpy())
        model_img   = Image.fromarray(model.output_layer.weight.data.numpy())
        output_img  = Image.fromarray(output.data.numpy())
        
        pred = output.data.max(1)[1] 
        correct = pred.eq(target.long()).sum()
        
        if plot:
            fig = plt.figure(1,(16., 16.))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1 , 3),
                             axes_pad=0.1)
    
            grid[0].matshow(dataset_img)
            grid[0].set_xlabel('DATASET', fontsize=24)
    
            grid[1].matshow(model_img)
            grid[1].set_xlabel('MODEL', fontsize=24)
            
            grid[2].matshow(output_img)
            grid[2].set_xlabel('OUTPUT', fontsize=24)
            
            plt.show()    
            
        print('correct: {}/{}, loss:{}'.format(correct, len(dataset), loss.data[0]))
            
        return dataset_img, model_img, output_img 

Lets take a closer look
~~~~~~~~~~~~~~~~~~~~~~~

with help from,
https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib

.. code:: python3

    import numpy
    fig = plt.figure(1, (80., 80.))
    grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 3),
                         axes_pad=0.1)
    
    
    data = [data.view(-1) for data, target in dataset]
    data = torch.stack(data)
    
    target = [target.view(-1) for data, target in dataset]
    target = torch.stack(target)
    
    grid[0].matshow(Image.fromarray(data.numpy()))
    grid[0].set_xlabel('DATASET', fontsize=72)
    for (x,y), val in numpy.ndenumerate(data.numpy()):
         grid[0].text(y, x, '{:d}'.format(int(val)), ha='center', va='center', fontsize=24,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    
            
    grid[1].matshow(Image.fromarray(model.output_layer.weight.data.numpy()))
    grid[1].set_xlabel('MODEL', fontsize=72)
    for (x,y), val in numpy.ndenumerate(model.output_layer.weight.data.numpy()):
         grid[1].text(y, x, '{:0.04f}'.format(val), ha='center', va='center',fontsize=16,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    
    output = model(Variable(data))
    grid[2].matshow(Image.fromarray(output.data.numpy()))
    grid[2].set_xlabel('OUTPUT', fontsize=72)
    for (x,y), val in numpy.ndenumerate(output.data.numpy()):
         grid[2].text(y, x, '{:0.04f}'.format(val), ha='center', va='center',fontsize=16,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    
    
    plt.show()



.. image::  /images/vanangamudimnist/output_19_0.png


.. code:: python3

    print(model(Variable(data[0].view(1, -1))))


.. parsed-literal::

    Variable containing:
    -2.6310 -2.2685 -2.7027 -1.6844 -2.3093 -2.8675 -2.0508 -2.2570 -2.9351 -2.0451
    [torch.FloatTensor of size 1x10]
    


.. code:: python3

    import numpy
    def plot_with_values(model, dataset):
        fig = plt.figure(1, (80., 80.))
        grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, 3),
                             axes_pad=0.5)
    
    
        data = [data.view(-1) for data, target in dataset]
        data = torch.stack(data)
    
        target = [target.view(-1) for data, target in dataset]
        target = torch.stack(target)
    
        grid[0].matshow(Image.fromarray(data.numpy()))
        grid[0].set_xlabel('DATASET', fontsize=144)
        for (x,y), val in numpy.ndenumerate(data.numpy()):
             grid[0].text(y, x, '{:d}'.format(int(val)), ha='center', va='center', fontsize=24,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    
    
        grid[1].matshow(Image.fromarray(model.output_layer.weight.data.numpy()))
        grid[1].set_xlabel('MODEL', fontsize=144)
        for (x,y), val in numpy.ndenumerate(model.output_layer.weight.data.numpy()):
             grid[1].text(y, x, '{:0.04f}'.format(val), ha='center', va='center',fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    
        output = model(Variable(data))
        grid[2].matshow(Image.fromarray(output.data.numpy()))
        grid[2].set_xlabel('OUTPUT', fontsize=144)
        for (x,y), val in numpy.ndenumerate(output.data.numpy()):
             grid[2].text(y, x, '{:0.04f}'.format(val), ha='center', va='center',fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    
    
        plt.show()

Before Training
~~~~~~~~~~~~~~~

.. code:: python3

    test_and_print(model, dataset)
    plot_with_values(model, dataset)



.. image::  /images/vanangamudimnist/output_23_0.png


.. parsed-literal::

    correct: 0/10, loss:2.4226558208465576



.. image::  /images/vanangamudimnist/output_23_2.png


Train the model once and see how it works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    train(model, optimizer, dataset)




.. parsed-literal::

    2.7791757822036742



.. code:: python3

    test_and_print(model, dataset)
    plot_with_values(model, dataset)



.. image::  /images/vanangamudimnist/output_26_0.png


.. parsed-literal::

    correct: 1/10, loss:2.178180694580078



.. image::  /images/vanangamudimnist/output_26_2.png


train once more and see the internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    train(model, optimizer, dataset)




.. parsed-literal::

    2.5523715019226074



.. code:: python3

    test_and_print(model, dataset)
    plot_with_values(model, dataset)



.. image::  /images/vanangamudimnist/output_29_0.png


.. parsed-literal::

    correct: 3/10, loss:2.002211093902588



.. image::  /images/vanangamudimnist/output_29_2.png


.. code:: python3

    print(data)
    print(model.output_layer.weight.data)
    print(output.data)


.. parsed-literal::

    
    
    Columns 0 to 12 
        0     0     1     1     1     0     0     1     0     1     0     0     1
        0     0     0     1     0     0     0     1     1     0     0     0     0
        0     0     1     1     1     0     0     0     0     1     0     0     1
        0     0     1     1     1     0     0     0     0     1     0     0     0
        0     0     1     0     1     0     0     1     0     1     0     0     1
        0     0     1     1     1     0     0     1     0     0     0     0     1
        0     0     1     1     1     0     0     1     0     0     0     0     1
        0     0     1     1     1     0     0     0     0     1     0     0     0
        0     0     1     1     1     0     0     1     0     1     0     0     1
        0     0     1     1     1     0     0     1     0     1     0     0     1
    
    Columns 13 to 24 
        0     1     0     0     1     0     1     0     0     1     1     1
        1     0     0     0     0     1     0     0     0     1     1     1
        1     1     0     0     1     0     0     0     0     1     1     1
        1     1     0     0     0     0     1     0     0     1     1     1
        1     1     0     0     0     0     1     0     0     0     0     1
        1     1     0     0     0     0     1     0     0     1     1     1
        1     1     0     0     1     0     1     0     0     1     1     1
        0     1     0     0     0     0     1     0     0     0     0     1
        1     1     0     0     1     0     1     0     0     1     1     1
        1     1     0     0     0     0     1     0     0     1     1     1
    [torch.FloatTensor of size 10x25]
    
    
    
    Columns 0 to 9 
    -0.0775 -0.0061 -0.0071  0.1463 -0.1230  0.0227  0.0463 -0.0465 -0.1198  0.0708
    -0.0960  0.1528 -0.0926  0.0864 -0.1663  0.0769 -0.0363  0.1534  0.1049 -0.1288
     0.0276 -0.0589  0.0924 -0.0903 -0.0308 -0.0125  0.0882 -0.2251 -0.0315  0.1383
     0.0111 -0.1083  0.0613 -0.0896 -0.0466  0.0821 -0.1129 -0.1794 -0.1222  0.1119
    -0.0691  0.1172 -0.0798 -0.2986  0.1729 -0.0716 -0.0129  0.0876 -0.0316  0.1776
    -0.0339 -0.1906 -0.1352  0.0487  0.0832  0.1772 -0.1550  0.0129  0.0819 -0.2758
    -0.1543 -0.1738  0.0953 -0.2080  0.1313 -0.0913  0.0670 -0.1560  0.1633 -0.0711
    -0.0094 -0.1027  0.0004 -0.0643  0.1243  0.1438  0.1580 -0.2372  0.1431  0.1654
     0.1362 -0.1573  0.1594 -0.0289  0.1687 -0.1899  0.1157 -0.0792 -0.1001 -0.0151
    -0.0201  0.0337 -0.0241  0.2104  0.1608  0.0809 -0.1552 -0.0117  0.1460  0.0480
    
    Columns 10 to 19 
    -0.1558  0.1090  0.0960 -0.1309 -0.0409 -0.0088  0.0410 -0.0089  0.0684  0.0250
     0.0668 -0.1753 -0.2620 -0.0575 -0.2509  0.1885 -0.1592 -0.0623  0.0243 -0.0514
     0.1202 -0.1700 -0.0461 -0.0950  0.0881 -0.0374  0.1558  0.1514  0.0030 -0.1964
     0.1443 -0.1058 -0.3234 -0.1575  0.0171  0.1277  0.1745  0.0623  0.1512  0.1205
    -0.1277  0.1327  0.0083  0.0360  0.1246 -0.0239 -0.1319 -0.1319  0.0878 -0.1434
     0.1563  0.1458  0.2015 -0.0692 -0.1107 -0.0947 -0.1483 -0.1704  0.1684  0.2312
    -0.1601  0.1189 -0.1619  0.1456  0.1256  0.0410  0.0905  0.1292 -0.0377  0.1647
     0.0471  0.0785  0.0429 -0.0644  0.0917 -0.1105  0.1714 -0.0471  0.0624  0.0284
     0.0962 -0.1139 -0.0221  0.2518  0.1174 -0.1687  0.0239  0.2354  0.1081 -0.1129
    -0.0703  0.1948  0.1462 -0.0811  0.0574 -0.0644 -0.1046 -0.2400  0.0926  0.2163
    
    Columns 20 to 24 
    -0.0508 -0.0765 -0.0687  0.0739 -0.2297
     0.0377 -0.0319  0.0110  0.2238 -0.0110
     0.1466  0.1324 -0.0583 -0.0821  0.1188
     0.1669  0.1309  0.1754 -0.0106  0.1148
    -0.1915  0.0448 -0.0192 -0.0364 -0.1001
    -0.0478 -0.0359  0.1435 -0.0016 -0.0797
     0.1716  0.0418  0.0949  0.0791 -0.0613
     0.1299  0.0025 -0.2672 -0.0257  0.0699
    -0.0256 -0.0980  0.0459 -0.1216  0.0568
     0.0782 -0.0537  0.2157  0.1004 -0.1848
    [torch.FloatTensor of size 10x25]
    
    
    -2.6310 -2.2685 -2.7027 -1.6844 -2.3093 -2.8675 -2.0508 -2.2570 -2.9351 -2.0451
    -2.3111 -2.5781 -2.8429 -2.1439 -2.1065 -2.3409 -2.2654 -2.1675 -2.5843 -1.9814
    -2.4300 -2.5531 -2.7617 -2.0206 -2.1601 -3.1099 -1.8916 -2.0229 -2.3853 -2.2703
    -2.4533 -2.4207 -2.7293 -2.0219 -2.3339 -3.0570 -1.6293 -2.3657 -2.6332 -2.1060
    -2.5164 -2.4800 -2.3811 -2.0388 -2.4169 -2.8624 -1.8470 -2.0271 -2.4557 -2.3934
    -2.3725 -2.3360 -2.8593 -2.0409 -2.3049 -2.6920 -2.0207 -2.2202 -2.6686 -1.9323
    -2.4916 -2.3202 -2.8539 -1.8623 -2.3450 -2.7756 -1.9968 -2.1851 -2.5846 -2.0858
    -2.5090 -2.3410 -2.2558 -1.9410 -2.5716 -2.8679 -1.8594 -2.4068 -2.5268 -2.1622
    -2.4983 -2.3660 -2.8174 -1.8228 -2.2903 -2.9784 -1.8904 -2.1410 -2.7288 -2.1487
    -2.3764 -2.3790 -2.8200 -1.9986 -2.2475 -2.8920 -1.9115 -2.1734 -2.8101 -1.9925
    [torch.FloatTensor of size 10x10]
    


Train over multiple epochs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python3

    def train_epochs(epochs, model, optim, dataset, print_every=100):
        snaps = []
        for epoch in range(epochs):
            avg_loss = train(model, optim, dataset)
            if not epoch % print_every:
                print('epoch: {}, loss:{}'.format(epoch, avg_loss/len(dataset)/10))
                snaps.append(test_and_print(model, dataset))
                
    
                return snaps

.. code:: python3

    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

.. code:: python3

    snaps = train_epochs(100, model, optimizer, dataset)


.. parsed-literal::

    epoch: 0, loss:0.024241248846054074



.. image::  /images/vanangamudimnist/output_34_1.png


.. parsed-literal::

    correct: 1/10, loss:2.3493340015411377


.. code:: python3

    fig = plt.figure(1, (16., 16.))
    grid = ImageGrid(fig, 111,
                         nrows_ncols=(len(snaps) , 3),
                         axes_pad=0.1)
    
    for i, snap in enumerate(snaps):
        for j, image in enumerate(snap):
            grid[i * 3 + j].matshow(image)
            
    grid[i * 3 + 0].set_xlabel('DATASET', fontsize=24)
    grid[i * 3 + 1].set_xlabel('MODEL', fontsize=24)
    grid[i * 3 + 2].set_xlabel('OUTPUT', fontsize=24)
            
    plt.show()



.. image::  /images/vanangamudimnist/output_35_0.png


.. code:: python3

    snaps = train_epochs(100000, model, optimizer, dataset, print_every=1000)


.. parsed-literal::

    epoch: 0, loss:1.4979378305724824e-05



.. image::  /images/vanangamudimnist/output_36_1.png


.. parsed-literal::

    correct: 10/10, loss:0.0014949440956115723
    epoch: 1000, loss:1.3566501729656011e-05



.. image::  /images/vanangamudimnist/output_36_3.png


.. parsed-literal::

    correct: 10/10, loss:0.0013541923835873604
    epoch: 2000, loss:1.2397395898005925e-05



.. image::  /images/vanangamudimnist/output_36_5.png


.. parsed-literal::

    correct: 10/10, loss:0.0012376849772408605
    epoch: 3000, loss:1.141426396497991e-05



.. image::  /images/vanangamudimnist/output_36_7.png


.. parsed-literal::

    correct: 10/10, loss:0.0011396838817745447
    epoch: 4000, loss:1.0574486601399258e-05



.. image::  /images/vanangamudimnist/output_36_9.png


.. parsed-literal::

    correct: 10/10, loss:0.0010559528600424528
    epoch: 5000, loss:9.851084818365054e-06



.. image::  /images/vanangamudimnist/output_36_11.png


.. parsed-literal::

    correct: 10/10, loss:0.0009838090045377612
    epoch: 6000, loss:9.220464067766443e-06



.. image::  /images/vanangamudimnist/output_36_13.png


.. parsed-literal::

    correct: 10/10, loss:0.000920907303225249
    epoch: 7000, loss:8.666477806400509e-06



.. image::  /images/vanangamudimnist/output_36_15.png


.. parsed-literal::

    correct: 10/10, loss:0.0008656416903249919
    epoch: 8000, loss:8.17526801256463e-06



.. image::  /images/vanangamudimnist/output_36_17.png


.. parsed-literal::

    correct: 10/10, loss:0.000816631130874157
    epoch: 9000, loss:7.73639925319003e-06



.. image::  /images/vanangamudimnist/output_36_19.png


.. parsed-literal::

    correct: 10/10, loss:0.0007728362688794732
    epoch: 10000, loss:7.342388962570113e-06



.. image::  /images/vanangamudimnist/output_36_21.png


.. parsed-literal::

    correct: 10/10, loss:0.000733515596948564
    epoch: 11000, loss:6.987177541304847e-06



.. image::  /images/vanangamudimnist/output_36_23.png


.. parsed-literal::

    correct: 10/10, loss:0.0006980619509704411
    epoch: 12000, loss:6.66436344909016e-06



.. image::  /images/vanangamudimnist/output_36_25.png


.. parsed-literal::

    correct: 10/10, loss:0.0006658405181951821
    epoch: 13000, loss:6.370135299221147e-06



.. image::  /images/vanangamudimnist/output_36_27.png


.. parsed-literal::

    correct: 10/10, loss:0.0006364684668369591
    epoch: 14000, loss:6.1013935264782045e-06



.. image::  /images/vanangamudimnist/output_36_29.png


.. parsed-literal::

    correct: 10/10, loss:0.0006096391589380801
    epoch: 15000, loss:5.854485207237304e-06



.. image::  /images/vanangamudimnist/output_36_31.png


.. parsed-literal::

    correct: 10/10, loss:0.0005849882145412266
    epoch: 16000, loss:5.626929625577759e-06



.. image::  /images/vanangamudimnist/output_36_33.png


.. parsed-literal::

    correct: 10/10, loss:0.0005622674943879247
    epoch: 17000, loss:5.416594656708185e-06



.. image::  /images/vanangamudimnist/output_36_35.png


.. parsed-literal::

    correct: 10/10, loss:0.0005412654136307538
    epoch: 18000, loss:5.2213049784768376e-06



.. image::  /images/vanangamudimnist/output_36_37.png


.. parsed-literal::

    correct: 10/10, loss:0.0005217638099566102
    epoch: 19000, loss:5.039620846218896e-06



.. image::  /images/vanangamudimnist/output_36_39.png


.. parsed-literal::

    correct: 10/10, loss:0.0005036209477111697
    epoch: 20000, loss:4.869873519055546e-06



.. image::  /images/vanangamudimnist/output_36_41.png


.. parsed-literal::

    correct: 10/10, loss:0.0004866681119892746
    epoch: 21000, loss:4.7111265157582235e-06



.. image::  /images/vanangamudimnist/output_36_43.png


.. parsed-literal::

    correct: 10/10, loss:0.00047081400407478213
    epoch: 22000, loss:4.562667345453519e-06



.. image::  /images/vanangamudimnist/output_36_45.png


.. parsed-literal::

    correct: 10/10, loss:0.00045598665019497275
    epoch: 23000, loss:4.423383987159469e-06



.. image::  /images/vanangamudimnist/output_36_47.png


.. parsed-literal::

    correct: 10/10, loss:0.0004420751647558063
    epoch: 24000, loss:4.292191806598566e-06



.. image::  /images/vanangamudimnist/output_36_49.png


.. parsed-literal::

    correct: 10/10, loss:0.0004289712815079838
    epoch: 25000, loss:4.168330851825886e-06



.. image::  /images/vanangamudimnist/output_36_51.png


.. parsed-literal::

    correct: 10/10, loss:0.00041659921407699585
    epoch: 26000, loss:4.051415649882984e-06



.. image::  /images/vanangamudimnist/output_36_53.png


.. parsed-literal::

    correct: 10/10, loss:0.00040492042899131775
    epoch: 27000, loss:3.9410730714735106e-06



.. image::  /images/vanangamudimnist/output_36_55.png


.. parsed-literal::

    correct: 10/10, loss:0.00039389816811308265
    epoch: 28000, loss:3.836599891656078e-06



.. image::  /images/vanangamudimnist/output_36_57.png


.. parsed-literal::

    correct: 10/10, loss:0.00038346173823811114
    epoch: 29000, loss:3.7375376195996066e-06



.. image::  /images/vanangamudimnist/output_36_59.png


.. parsed-literal::

    correct: 10/10, loss:0.0003735655336640775
    epoch: 30000, loss:3.643752643256448e-06



.. image::  /images/vanangamudimnist/output_36_61.png


.. parsed-literal::

    correct: 10/10, loss:0.00036419619573280215
    epoch: 31000, loss:3.554246144631179e-06



.. image::  /images/vanangamudimnist/output_36_63.png


.. parsed-literal::

    correct: 10/10, loss:0.00035525442217476666
    epoch: 32000, loss:3.4688986270339234e-06



.. image::  /images/vanangamudimnist/output_36_65.png


.. parsed-literal::

    correct: 10/10, loss:0.0003467276110313833
    epoch: 33000, loss:3.38783597908332e-06



.. image::  /images/vanangamudimnist/output_36_67.png


.. parsed-literal::

    correct: 10/10, loss:0.0003386292955838144
    epoch: 34000, loss:3.3104618269135243e-06



.. image::  /images/vanangamudimnist/output_36_69.png


.. parsed-literal::

    correct: 10/10, loss:0.00033089861972257495
    epoch: 35000, loss:3.236417862353846e-06



.. image::  /images/vanangamudimnist/output_36_71.png


.. parsed-literal::

    correct: 10/10, loss:0.0003235008043702692
    epoch: 36000, loss:3.1655030616093424e-06



.. image::  /images/vanangamudimnist/output_36_73.png


.. parsed-literal::

    correct: 10/10, loss:0.00031641527311876416
    epoch: 37000, loss:3.097559627349256e-06



.. image::  /images/vanangamudimnist/output_36_75.png


.. parsed-literal::

    correct: 10/10, loss:0.0003096264263149351
    epoch: 38000, loss:3.0326169580803253e-06



.. image::  /images/vanangamudimnist/output_36_77.png


.. parsed-literal::

    correct: 10/10, loss:0.00030313775641843677
    epoch: 39000, loss:2.9704941298405175e-06



.. image::  /images/vanangamudimnist/output_36_79.png


.. parsed-literal::

    correct: 10/10, loss:0.00029693052056245506
    epoch: 40000, loss:2.9106522124493495e-06



.. image::  /images/vanangamudimnist/output_36_81.png


.. parsed-literal::

    correct: 10/10, loss:0.0002909509348683059
    epoch: 41000, loss:2.8533561817312146e-06



.. image::  /images/vanangamudimnist/output_36_83.png


.. parsed-literal::

    correct: 10/10, loss:0.0002852260076906532
    epoch: 42000, loss:2.797896486299578e-06



.. image::  /images/vanangamudimnist/output_36_85.png


.. parsed-literal::

    correct: 10/10, loss:0.00027968379436060786
    epoch: 43000, loss:2.744394249020843e-06



.. image::  /images/vanangamudimnist/output_36_87.png


.. parsed-literal::

    correct: 10/10, loss:0.0002743378863669932
    epoch: 44000, loss:2.6928670158667955e-06



.. image::  /images/vanangamudimnist/output_36_89.png


.. parsed-literal::

    correct: 10/10, loss:0.0002691886038519442
    epoch: 45000, loss:2.6431842416059228e-06



.. image::  /images/vanangamudimnist/output_36_91.png


.. parsed-literal::

    correct: 10/10, loss:0.00026422421797178686
    epoch: 46000, loss:2.595654157630634e-06



.. image::  /images/vanangamudimnist/output_36_93.png


.. parsed-literal::

    correct: 10/10, loss:0.00025947438552975655
    epoch: 47000, loss:2.5499546827632003e-06



.. image::  /images/vanangamudimnist/output_36_95.png


.. parsed-literal::

    correct: 10/10, loss:0.0002549075579736382
    epoch: 48000, loss:2.5057809216377793e-06



.. image::  /images/vanangamudimnist/output_36_97.png


.. parsed-literal::

    correct: 10/10, loss:0.00025049326359294355
    epoch: 49000, loss:2.4633905304654037e-06



.. image::  /images/vanangamudimnist/output_36_99.png


.. parsed-literal::

    correct: 10/10, loss:0.0002462573756929487
    epoch: 50000, loss:2.4225734814535825e-06



.. image::  /images/vanangamudimnist/output_36_101.png


.. parsed-literal::

    correct: 10/10, loss:0.00024217806640081108
    epoch: 51000, loss:2.3830521495256107e-06



.. image::  /images/vanangamudimnist/output_36_103.png


.. parsed-literal::

    correct: 10/10, loss:0.0002382286766078323
    epoch: 52000, loss:2.344911696127383e-06



.. image::  /images/vanangamudimnist/output_36_105.png


.. parsed-literal::

    correct: 10/10, loss:0.0002344170061405748
    epoch: 53000, loss:2.3082216066541153e-06



.. image::  /images/vanangamudimnist/output_36_107.png


.. parsed-literal::

    correct: 10/10, loss:0.00023075027274899185
    epoch: 54000, loss:2.272542646096554e-06



.. image::  /images/vanangamudimnist/output_36_109.png


.. parsed-literal::

    correct: 10/10, loss:0.00022718461696058512
    epoch: 55000, loss:2.237642715044785e-06



.. image::  /images/vanangamudimnist/output_36_111.png


.. parsed-literal::

    correct: 10/10, loss:0.00022369690123014152
    epoch: 56000, loss:2.2039403484086506e-06



.. image::  /images/vanangamudimnist/output_36_113.png


.. parsed-literal::

    correct: 10/10, loss:0.000220328540308401
    epoch: 57000, loss:2.171287556848256e-06



.. image::  /images/vanangamudimnist/output_36_115.png


.. parsed-literal::

    correct: 10/10, loss:0.00021706517145503312
    epoch: 58000, loss:2.139623753464548e-06



.. image::  /images/vanangamudimnist/output_36_117.png


.. parsed-literal::

    correct: 10/10, loss:0.0002139005810022354
    epoch: 59000, loss:2.108841326844413e-06



.. image::  /images/vanangamudimnist/output_36_119.png


.. parsed-literal::

    correct: 10/10, loss:0.00021082404418848455
    epoch: 60000, loss:2.0792475115740673e-06



.. image::  /images/vanangamudimnist/output_36_121.png


.. parsed-literal::

    correct: 10/10, loss:0.0002078662655549124
    epoch: 61000, loss:2.0501944491115864e-06



.. image::  /images/vanangamudimnist/output_36_123.png


.. parsed-literal::

    correct: 10/10, loss:0.00020496267825365067
    epoch: 62000, loss:2.0219945909047967e-06



.. image::  /images/vanangamudimnist/output_36_125.png


.. parsed-literal::

    correct: 10/10, loss:0.0002021441760007292
    epoch: 63000, loss:1.9942749677284153e-06



.. image::  /images/vanangamudimnist/output_36_127.png


.. parsed-literal::

    correct: 10/10, loss:0.0001993737678276375
    epoch: 64000, loss:1.9675384701258736e-06



.. image::  /images/vanangamudimnist/output_36_129.png


.. parsed-literal::

    correct: 10/10, loss:0.00019670158508233726
    epoch: 65000, loss:1.9416159211687046e-06



.. image::  /images/vanangamudimnist/output_36_131.png


.. parsed-literal::

    correct: 10/10, loss:0.0001941107475431636
    epoch: 66000, loss:1.9162728531227913e-06



.. image::  /images/vanangamudimnist/output_36_133.png


.. parsed-literal::

    correct: 10/10, loss:0.00019157768110744655
    epoch: 67000, loss:1.8914568463515025e-06



.. image::  /images/vanangamudimnist/output_36_135.png


.. parsed-literal::

    correct: 10/10, loss:0.00018909727805294096
    epoch: 68000, loss:1.8671469733817505e-06



.. image::  /images/vanangamudimnist/output_36_137.png


.. parsed-literal::

    correct: 10/10, loss:0.00018666766118258238
    epoch: 69000, loss:1.8432843062328176e-06



.. image::  /images/vanangamudimnist/output_36_139.png


.. parsed-literal::

    correct: 10/10, loss:0.00018428244220558554
    epoch: 70000, loss:1.8202659266535192e-06



.. image::  /images/vanangamudimnist/output_36_141.png


.. parsed-literal::

    correct: 10/10, loss:0.00018198174075223505
    epoch: 71000, loss:1.7980593765969386e-06



.. image::  /images/vanangamudimnist/output_36_143.png


.. parsed-literal::

    correct: 10/10, loss:0.00017976219533011317
    epoch: 72000, loss:1.776098020854988e-06



.. image::  /images/vanangamudimnist/output_36_145.png


.. parsed-literal::

    correct: 10/10, loss:0.00017756715533323586
    epoch: 73000, loss:1.7548067298776003e-06



.. image::  /images/vanangamudimnist/output_36_147.png


.. parsed-literal::

    correct: 10/10, loss:0.00017543925787322223
    epoch: 74000, loss:1.7340696340397697e-06



.. image::  /images/vanangamudimnist/output_36_149.png


.. parsed-literal::

    correct: 10/10, loss:0.00017336638120468706
    epoch: 75000, loss:1.7136688547907397e-06



.. image::  /images/vanangamudimnist/output_36_151.png


.. parsed-literal::

    correct: 10/10, loss:0.00017132725042756647
    epoch: 76000, loss:1.6937134751060513e-06



.. image::  /images/vanangamudimnist/output_36_153.png


.. parsed-literal::

    correct: 10/10, loss:0.00016933264851104468
    epoch: 77000, loss:1.674388626270229e-06



.. image::  /images/vanangamudimnist/output_36_155.png


.. parsed-literal::

    correct: 10/10, loss:0.00016740091086830944
    epoch: 78000, loss:1.6554753019590862e-06



.. image::  /images/vanangamudimnist/output_36_157.png


.. parsed-literal::

    correct: 10/10, loss:0.00016551054432056844
    epoch: 79000, loss:1.637008828765829e-06



.. image::  /images/vanangamudimnist/output_36_159.png


.. parsed-literal::

    correct: 10/10, loss:0.0001636647357372567
    epoch: 80000, loss:1.6191334980248941e-06



.. image::  /images/vanangamudimnist/output_36_161.png


.. parsed-literal::

    correct: 10/10, loss:0.00016187792061828077
    epoch: 81000, loss:1.6014868397178361e-06



.. image::  /images/vanangamudimnist/output_36_163.png


.. parsed-literal::

    correct: 10/10, loss:0.00016011403931770474
    epoch: 82000, loss:1.584194154929719e-06



.. image::  /images/vanangamudimnist/output_36_165.png


.. parsed-literal::

    correct: 10/10, loss:0.00015838549006730318
    epoch: 83000, loss:1.5670943212171552e-06



.. image::  /images/vanangamudimnist/output_36_167.png


.. parsed-literal::

    correct: 10/10, loss:0.00015667623665649444
    epoch: 84000, loss:1.5502425030717862e-06



.. image::  /images/vanangamudimnist/output_36_169.png


.. parsed-literal::

    correct: 10/10, loss:0.0001549917069496587
    epoch: 85000, loss:1.5339244127972053e-06



.. image::  /images/vanangamudimnist/output_36_171.png


.. parsed-literal::

    correct: 10/10, loss:0.00015336077194660902
    epoch: 86000, loss:1.5179560068645515e-06



.. image::  /images/vanangamudimnist/output_36_173.png


.. parsed-literal::

    correct: 10/10, loss:0.0001517643395345658
    epoch: 87000, loss:1.5023343512439169e-06



.. image::  /images/vanangamudimnist/output_36_175.png


.. parsed-literal::

    correct: 10/10, loss:0.00015020293358247727
    epoch: 88000, loss:1.4870688719383907e-06



.. image::  /images/vanangamudimnist/output_36_177.png


.. parsed-literal::

    correct: 10/10, loss:0.00014867704885546118
    epoch: 89000, loss:1.4721054612891748e-06



.. image::  /images/vanangamudimnist/output_36_179.png


.. parsed-literal::

    correct: 10/10, loss:0.00014718130114488304
    epoch: 90000, loss:1.4575626482837834e-06



.. image::  /images/vanangamudimnist/output_36_181.png


.. parsed-literal::

    correct: 10/10, loss:0.00014572765212506056
    epoch: 91000, loss:1.44314118733746e-06



.. image::  /images/vanangamudimnist/output_36_183.png


.. parsed-literal::

    correct: 10/10, loss:0.00014428579015657306
    epoch: 92000, loss:1.4288767615653342e-06



.. image::  /images/vanangamudimnist/output_36_185.png


.. parsed-literal::

    correct: 10/10, loss:0.00014286009536590427
    epoch: 93000, loss:1.4148931950330734e-06



.. image::  /images/vanangamudimnist/output_36_187.png


.. parsed-literal::

    correct: 10/10, loss:0.0001414622092852369
    epoch: 94000, loss:1.4011994953762042e-06



.. image::  /images/vanangamudimnist/output_36_189.png


.. parsed-literal::

    correct: 10/10, loss:0.00014009341248311102
    epoch: 95000, loss:1.3878234531148337e-06



.. image::  /images/vanangamudimnist/output_36_191.png


.. parsed-literal::

    correct: 10/10, loss:0.00013875641161575913
    epoch: 96000, loss:1.3746419735980453e-06



.. image::  /images/vanangamudimnist/output_36_193.png


.. parsed-literal::

    correct: 10/10, loss:0.0001374386774841696
    epoch: 97000, loss:1.361803597319522e-06



.. image::  /images/vanangamudimnist/output_36_195.png


.. parsed-literal::

    correct: 10/10, loss:0.00013615534408017993
    epoch: 98000, loss:1.3493407550413395e-06



.. image::  /images/vanangamudimnist/output_36_197.png


.. parsed-literal::

    correct: 10/10, loss:0.00013490939454641193
    epoch: 99000, loss:1.337103256446426e-06



.. image::  /images/vanangamudimnist/output_36_199.png


.. parsed-literal::

    correct: 10/10, loss:0.0001336860441369936


.. code:: python3

    torch.save(model.state_dict(), 'model_150000.pth')

.. code:: python3

    plot_with_values(model, dataset)



.. image::  /images/vanangamudimnist/output_38_0.png


.. code:: python3

    normalized_model = model.output_layer.weight.data.numpy()
    normalized_model = numpy.absolute(normalized_model)
    total  =  normalized_model.sum()
    normalized_model = normalized_model/total
    
    plt.matshow(normalized_model)




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7f12dd850b38>




.. image::  /images/vanangamudimnist/output_39_1.png

