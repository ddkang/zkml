// Generics
pub mod averager;

pub mod arithmetic;
pub mod shape;

// Concrete implementations
pub mod avg_pool_2d;
pub mod batch_mat_mul;
pub mod conv2d;
pub mod fully_connected;
pub mod logistic;
pub mod mean;
pub mod noop;
pub mod rsqrt;
pub mod softmax;
pub mod squared_diff;

// Special: dag
pub mod dag;

// Special: layer
pub mod layer;
