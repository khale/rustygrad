use std::ops;
use std::fmt;
use std::iter;

// who produced me
// TODO: change to state pattern: 
#[derive(Clone, Copy)]
pub enum OpType {
    NoOp,
    Add,
    Mul,
    Pow,
    ReLU,
}

impl fmt::Debug for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            OpType::NoOp => write!(f, "<None>"),
            OpType::Add  => write!(f, "+"),
            OpType::Mul  => write!(f, "*"),
            OpType::Pow  => write!(f, "**"),
            OpType::ReLU => write!(f, "ReLU"),
        }
    }
}

// this will eventually become a Tensor
#[derive(Clone)]
pub struct Value {
    data: f64,
    grad: f64,
    constant: f64,
    prod_op: OpType,
    producers: Vec<Value>,
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("prod_op", &self.prod_op)
            .field("producers", &self.producers)
            .finish()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Value) -> bool {
        self.data == other.data && 
        self.grad == other.grad
    }
}

#[derive(Clone)]
pub struct ValueBuilder {
    data: f64,
    constant: f64, // only applicable for unary ops with constants (pow)
    grad: f64,
    producers: Vec<Value>,
    prod_op: OpType,
}

impl ValueBuilder {
    pub fn new() -> Self {
        ValueBuilder { 
            data: 0.0, 
            grad: 0.0, 
            constant: 0.0,
            producers: Vec::new(),
            prod_op: OpType::NoOp,
        } 
    }

    pub fn data(&mut self, data: f64) -> &mut Self {
        self.data = data;
        self
    }

    pub fn grad(&mut self, grad: f64) -> &mut Self {
        self.grad = grad;
        self
    }

    pub fn prod_op(&mut self, prod_op: OpType) -> &mut Self {
        self.prod_op = prod_op;
        self
    }

    pub fn producer(&mut self, prod: Value) -> &mut Self {
        self.producers.push(prod);
        self
    }

    pub fn constant(&mut self, constant: f64) -> &mut Self {
        self.constant = constant;
        self
    }

    pub fn finalize(&mut self) -> Value {
        Value { 
            data: self.data, 
            grad: self.grad, 
            constant: self.constant,
            prod_op: self.prod_op.clone(),
            producers: self.producers.clone(),
        }
    }
}

// this computes the gradient for the (unary or binary) inputs of an operation with respect
// to the final node in the expr tree
fn compute_grad(v: &Value) -> (f64, Option<f64>) {
    match v.prod_op {
        OpType::NoOp => { panic!("Should not compute gradient on leaf node!\n"); }
        OpType::Add  => (v.grad, Some(v.grad)), 
            // g = a + b; 
            // df/da = df/dg * dg/da = df/dg * 1;
            // df/db = df/dg * dg/db = df/dg * 1;
        OpType::Mul  => {
            let a = v.producers[0].data;
            let b = v.producers[1].data;
            (b*v.grad, Some(a*v.grad))
                // g = a * b;
                // df/da = df/dg * dg/da = df/dg * b
                // df/db = df/dg * dg/db = df/dg * a
        }
        OpType::Pow  => {
            let k = v.constant;
            let x = v.producers[0].data;
            (k * x.powf(k-1.0) * v.grad, None)
                // g = x^k
                // df/dx = df/dg * dg/dx = df/dg * kx^(k-1)
        }
        OpType::ReLU => {
            let grad = {
                if v.data > 0.0 {
                    v.grad
                } else {
                    0.0
                }
            };
            (grad, None)
        }
    }
}

fn _backward(v: &mut Value, mygrad: f64) {
    v.grad += mygrad;

    // base case: leaf node
    if v.producers.is_empty() {
        return
    }

    let grads = compute_grad(&v);

    _backward(&mut v.producers[0], grads.0);

    // binary op
    if v.producers.len() > 1 {
        _backward(&mut v.producers[1], grads.1.unwrap());
    }
}

fn _zero_grads(v: &mut Value) {
    v.grad = 0.0;
    for p in v.producers.iter_mut() {
        _zero_grads(p);
    }
}

impl Value {
    pub fn relu(self) -> Self {
        ValueBuilder::new()
            .data(if self.data < 0.0 { 0.0 } else { self.data })
            .prod_op(OpType::ReLU)
            .producer(self)
            .finalize()
    }

    pub fn backward(&mut self) {
        _backward(self, 1.0);
    }

    pub fn zero_gradients(&mut self) {
        _zero_grads(self);
    }


    // should give a pre-order traversal
    pub fn values<'a>(&'a self) -> Box<dyn Iterator<Item = &Value> + 'a> {
        Box::new(
            iter::once(self)
                .chain(self.producers.iter().map(|n| n.values()).flatten()),
        )
    }

    pub fn check_gradients(&self, expected: &[f64]) -> bool {
        for (i, v) in self.values().enumerate() {
            if v.grad != expected[i] {
               return false;
            }
        }
        true
    }

    pub fn print_grads(&self) {
        for i in self.values() {
            print!("{:?} ", i.grad);
        }
        println!("");
    }

}

pub trait Pow<Rhs = Self> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}

// KCH: Only constant powers for now
//impl Pow<Value> for Value {
    //type Output = Self;
    //fn pow(self, other: Self) -> Self {
        //ValueBuilder::new()
            //.data(self.data.powf(other.data))
            //.prod_op(OpType::Pow)
            //.producer(self)
            //.producer(other)
            //.finalize()
    //}
//}

impl Pow<f64> for Value {
    type Output = Self;
    fn pow(self, other: f64) -> Self {

        ValueBuilder::new()
            .data(self.data.powf(other))
            .prod_op(OpType::Pow)
            .producer(self)
            .constant(other)
            .finalize()
    }
}

impl ops::Add<f64> for Value {
    type Output = Self;
    fn add(self, other: f64) -> Self {
        let o = ValueBuilder::new()
            .data(other)
            .finalize();

        ValueBuilder::new()
            .data(self.data + o.data)
            .prod_op(OpType::Add)
            .producer(self)
            .producer(o)
            .finalize()
    }
}

impl ops::Add<Value> for f64 {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        let me = ValueBuilder::new()
            .data(self)
            .finalize();

        ValueBuilder::new()
            .data(self + other.data)
            .prod_op(OpType::Add)
            .producer(me)
            .producer(other)
            .finalize()
    }
}

impl ops::Add<Value> for Value {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        ValueBuilder::new()
            .data(self.data + other.data)
            .prod_op(OpType::Add)
            .producer(self)
            .producer(other)
            .finalize()
    }
}

impl ops::Sub<Value> for Value {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl ops::Sub<Value> for f64 {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        let me = ValueBuilder::new()
            .data(self)
            .finalize();

        me + (-other)
    }
}

impl ops::Sub<f64> for Value {
    type Output = Self;
    fn sub(self, other: f64) -> Self {
        self + (-other)
    }
}

impl ops::Mul<Value> for Value {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        ValueBuilder::new()
            .data(self.data * other.data)
            .prod_op(OpType::Mul)
            .producer(self)
            .producer(other)
            .finalize()
    }
}

impl ops::Mul<f64> for Value {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        let o = ValueBuilder::new()
            .data(other)
            .finalize();

        ValueBuilder::new()
            .data(self.data * o.data)
            .prod_op(OpType::Mul)
            .producer(self)
            .producer(o)
            .finalize()
    }
}

impl ops::Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        let me = ValueBuilder::new()
            .data(self)
            .finalize();

        ValueBuilder::new()
            .data(self * other.data)
            .prod_op(OpType::Mul)
            .producer(me)
            .producer(other)
            .finalize()
    }
}

impl ops::Div<Value> for Value {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self * other.pow(-1.0)
    }
}

impl ops::Div<f64> for Value {
    type Output = Self;
    fn div(self, other: f64) -> Self {
        let o = ValueBuilder::new()
            .data(other)
            .finalize();
        
        self * o.pow(-1.0)
    }
}

impl ops::Div<Value> for f64 {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        let me = ValueBuilder::new()
            .data(self)
            .finalize();

        me * other.pow(-1.0)
    }
}

impl ops::Neg for Value {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        let v2 = ValueBuilder::new().data(30.0).finalize();
        assert_eq!(v1 + v2, ValueBuilder::new().data(50.0).finalize());
    }

    #[test]
    fn adds_float_left_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        assert_eq!(v1 + 30.0, ValueBuilder::new().data(50.0).finalize());
    }

    #[test]
    fn adds_float_right_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        assert_eq!(30.0 + v1, ValueBuilder::new().data(50.0).finalize());
    }

    #[test]
    fn subs_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        let v2 = ValueBuilder::new().data(10.0).finalize();
        assert_eq!(v1 - v2, ValueBuilder::new().data(10.0).finalize());
    }

    #[test]
    fn subs_float_left_correctly() {
        let v1 = ValueBuilder::new().data(10.0).finalize();
        assert_eq!(20.0 - v1, ValueBuilder::new().data(10.0).finalize());
    }

    #[test]
    fn subs_float_right_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        assert_eq!(v1 - 10.0, ValueBuilder::new().data(10.0).finalize());
    }

    #[test]
    fn muls_correctly() {
        let v1 = ValueBuilder::new().data(-20.0).finalize();
        let v2 = ValueBuilder::new().data(10.0).finalize();
        assert_eq!(v1 * v2, ValueBuilder::new().data(-200.0).finalize());
    }

    #[test]
    fn muls_float_left_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        assert_eq!(10.0 * v1, ValueBuilder::new().data(200.0).finalize());
    }

    #[test]
    fn muls_float_right_correctly() {
        let v1 = ValueBuilder::new().data(10.0).finalize();
        assert_eq!(v1 * 20.0, ValueBuilder::new().data(200.0).finalize());
    }

    #[test]
    fn divs_correctly() {
        let v1 = ValueBuilder::new().data(200.0).finalize();
        let v2 = ValueBuilder::new().data(10.0).finalize();
        assert_eq!(v1 / v2, ValueBuilder::new().data(20.0).finalize());
    }

    #[test]
    fn divs_float_left_correctly() {
        let v1 = ValueBuilder::new().data(20.0).finalize();
        assert_eq!(200.0 / v1, ValueBuilder::new().data(10.0).finalize());
    }

    #[test]
    fn divs_float_right_correctly() {
        let v1 = ValueBuilder::new().data(100.0).finalize();
        assert_eq!(v1 / 20.0, ValueBuilder::new().data(5.0).finalize());
    }

    #[test]
    fn negs_correctly() {
        let v1 = ValueBuilder::new().data(-20.0).finalize();
        assert_eq!(-v1, ValueBuilder::new().data(20.0).finalize());
    }

    #[test] 
    fn pows_correctly() {
        let v1 = ValueBuilder::new().data(10.0).finalize();
        assert_eq!(v1.pow(2.0), ValueBuilder::new().data(100.0).finalize());
    }

    #[test]
    fn relus_correctly() {
        let v1 = ValueBuilder::new().data(-2.0).finalize();
        let v2 = ValueBuilder::new().data(3.0).finalize();
        assert_eq!(v1.relu(), ValueBuilder::new().data(0.0).finalize());
        assert_eq!(v2.relu(), ValueBuilder::new().data(3.0).finalize());
    }
    
    #[test]
    fn simple_add_grads_correctly() {
        let v1 = ValueBuilder::new().data(-2.0).finalize();
        let v2 = ValueBuilder::new().data(3.0).finalize();
        let mut v3 = v1 + v2;
        v3.backward();
        assert_eq!(true, v3.check_gradients(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn simple_mul_grads_correctly() {
        let v1 = ValueBuilder::new().data(-2.0).finalize();
        let v2 = ValueBuilder::new().data(3.0).finalize();
        let mut v3 = v1 * v2;
        v3.backward();
        assert_eq!(true, v3.check_gradients(&[1.0, 3.0, -2.0]));
    }

    #[test]
    fn simple_pow_grads_correctly() {
        let v1 = ValueBuilder::new().data(10.0).finalize();
        let mut v2 = v1.pow(2.0);
        v2.backward();
        assert_eq!(true, v2.check_gradients(&[1.0, 20.0]));
    }

    #[test]
    fn simple_relu_grads_correctly_pos() {
        let v1 = ValueBuilder::new().data(10.0).finalize();
        let mut v2 = v1.relu();
        v2.backward();
        assert_eq!(true, v2.check_gradients(&[1.0, 1.0]));
    }

    #[test]
    fn simple_relu_grads_correctly_neg() {
        let v1 = ValueBuilder::new().data(-10.0).finalize();
        let mut v2 = v1.relu();
        v2.backward();
        assert_eq!(true, v2.check_gradients(&[1.0, 0.0]));
    }

    #[test]
    fn linear_2level_expr_grads_correctly() {
        let v1 = ValueBuilder::new().data(2.0).finalize();
        let v2 = ValueBuilder::new().data(3.0).finalize();
        let v3 = v1 + v2;
        let mut v4 = v3 * 4.0;
        v4.backward();
        assert_eq!(true, v4.check_gradients(&[1.0, 4.0, 4.0, 4.0, 5.0]));
    }

    #[test]
    fn nonlinear_2level_expr_grads_correctly() {
        let v1 = ValueBuilder::new().data(-2.0).finalize();
        let v2 = ValueBuilder::new().data(3.0).finalize();
        let v3 = v1 + v2;
        let v4 = v3 * 4.0;
        let mut v5 = v4.relu();
        v5.backward();
        assert_eq!(true, v5.check_gradients(&[1.0, 1.0, 4.0, 4.0, 4.0, 1.0]));
    }

    #[test]
    fn nonlinear_2level_expr_grads_correctly_neg() {
        let v1 = ValueBuilder::new().data(2.0).finalize();
        let v2 = ValueBuilder::new().data(3.0).finalize();
        let v3 = v1 + v2;
        let v4 = v3 * -4.0;
        let mut v5 = v4.relu();
        v5.backward();
        v5.print_grads();
        assert_eq!(true, v5.check_gradients(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
    }
}
