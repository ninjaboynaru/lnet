
# LNET
Convocational neural network implemented in Go as a learning project.  
Built in order to test my understanding after reading ["Neural Networks From Scratch"](https://nnfs.io/) by [Setndex](https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ).  

## Barebones Approach
Where as most neural network implementations use a matrix of weight values and a vector of biases to represent a layer, I opted to separate neurons and layers into their own abstractions/structs.  

That is, in my implementation a `layer` owns many `neurons` and a `neuron` owns a `bias` value and an array of `weights`. `input` is passed to a `layer` which then passed it to each of its `neurons` to be processed by them.  

This approach is less performant and results in more code than the traditional approach does, but forces one to attain a deeper understanding of each and every operation that occurs at every level and phase of the network.