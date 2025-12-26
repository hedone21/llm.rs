#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            0
        } else {
            self.dims.iter().product()
        }
    }
}
