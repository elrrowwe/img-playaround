# img playaround

With my interest in GANs soaring as of recently, I have been curious about how certain image processing algorithms work, and whether I could implement them. So, finally, I got around actually trying, practicing, experimenting and in general having fun with the aforementioned algorithms. This repo will contain my experiments with these algorithms, including some relevant tasks I had to do for university. 

# canny edge detection 
The file imgplayround.py includes my implementation of the Canny edge detection algorithm (https://en.wikipedia.org/wiki/Canny_edge_detector), with the dependencies being numpy (trigonometry, operations on matrices etc.) PIL (opening files, converting back and forth between images and arrays, showing the images) and scipy (convolutions). 
The function does a reasonably good job in detecting edges in non incredibly noisy images. However, the implementation will be further improved by experimenting with possible optimization techniques.


Before edge detection:


![politechnikalodzka](https://github.com/elrrowwe/img-playaround/assets/116558151/074c2370-aea2-4495-a1c3-6d6dc1cbcfa1)

After edge detection: 

<img width="930" alt="edges_pl" src="https://github.com/elrrowwe/img-playaround/assets/116558151/7ca8fde6-a56f-400f-8982-3692ea7fb8dd">
