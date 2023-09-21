# RustyGrad
Dead simple autograd library in Rust based on Andrej Karpathy's [micrograd](https://youtu.be/VMj-3S1tku0) engine. 
Really just a weekend project to learn more Rust and DNN concepts, but I might try to turn it into something more
useful. 

- Scalar values only
- Only nonlinearity supported is ReLU
- No parallelization whatsoever

## Ugliness
The most annoying thing about this implementation currently is that the
backward pass is very clunky. I'd prefer a functional-style implementation
where the gradient functions are captured in closures bound when values are
created, but I couldn't figure out how to do this in Rust without upsetting the
borrow checker. I suspect you could do something similar with traits using a [state
pattern](https://doc.rust-lang.org/book/ch17-03-oo-design-patterns.html), but
it seemed like overkill at the time. 

## TODOs
Maybe if I have time at some point:
[ ] More comprehensive testing for backward passes
[ ] Cleaner organization
[ ] Functional-style gradient functions for the various operators
[ ] Support for Tensors
[ ] Parallelism
[ ] Softmax, tanh, etc. 

## References
- `Vec`-based [graph representations in Rust](https://paulkernfeld.com/2018/06/17/exploring-computation-graphs-in-rust.html)
