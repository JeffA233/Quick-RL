// use std::{
//     fmt::{Display, Debug},
//     ops::{Deref, DerefMut}
// };

use tch::Tensor;

pub fn print_tensor_vecf32(ten_name: &str, ten: &Tensor) {
    println!(
        "Tensor {ten_name} {{\n   size: {:#?},\n  device: {:#?},\n    val: {:#?},\n}}\n",
        ten.size(),
        ten.device(),
        Vec::<f32>::try_from(ten).unwrap(),
    );
}

pub fn print_tensor_noval(ten_name: &str, ten: &Tensor) {
    println!(
        "Tensor {ten_name} {{\n   size: {:#?},\n  device: {:#?},\n\n",
        ten.size(),
        ten.device(),
        // Vec::<f32>::try_from(ten).unwrap(),
    );
}

pub fn print_tensor_2df32(ten_name: &str, ten: &Tensor) {
    println!(
        "Tensor {ten_name} {{\n   size: {:#?},\n  device: {:#?},\n    val: {:#?},\n}}\n",
        ten.size(),
        ten.device(),
        Vec::<Vec<f32>>::try_from(ten).unwrap(),
    );
}

pub fn print_tensor_3df32(ten_name: &str, ten: &Tensor) {
    println!(
        "Tensor {ten_name} {{\n   size: {:#?},\n  device: {:#?},\n    val: {:#?},\n}}\n",
        ten.size(),
        ten.device(),
        Vec::<Vec<Vec<f32>>>::try_from(ten).unwrap(),
    );
}

// pub fn print_tensor_4df32(ten_name: &str, ten: &Tensor) {
//     println!("Tensor {ten_name} {{\n   size: {:#?},\n  device: {:#?},\n    val: {:#?},\n}}\n",
//         ten.size(),
//         ten.device(),
//         Vec::<Vec<Vec<Vec<f32>>>>::try_from(ten).unwrap(),
//     );
// }

pub fn print_tensor_value(ten: &Tensor) {
    println!("{}", ten);
}

// #[debugger_visualizer(natvis_file = "TensorDisplay.natvis")]
// pub struct TensorDisplay(pub Tensor);

// #[debugger_visualizer(natvis_file = "TensorDisplay.xml")]
// #[debugger_visualizer(gdb_script_file = "tensor_display.py")]
// pub mod ten_display {
//     use std::{fmt::{Display, Debug}, ops::{Deref, DerefMut}};

//     use tch::{Kind, Tensor, Device};

//     pub struct TensorDisplay {
//         tensor: Tensor,
//         size: Vec<i64>,
//         kind: Kind,
//         max: f64,
//         min: f64,
//         device: Device,
//     }

//     impl TensorDisplay {
//         pub fn new(ten: Tensor) -> Self {
//             let size = ten.size();
//             let kind = ten.kind();
//             let max = f64::try_from(ten.max()).unwrap();
//             let min = f64::try_from(ten.min()).unwrap();
//             let device = ten.device();
//             Self {
//                 tensor: ten,
//                 size,
//                 kind,
//                 max,
//                 min,
//                 device,
//             }
//         }
//     }

//     impl Display for TensorDisplay {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             write!(f, "
//             Tensor {{\n
//                 size: {:#?},\n
//                 device: {:#?},\n
//             }}
//             ",
//             // self.tensor.size(),
//             self.size,
//             // self.tensor.device(),)
//             self.device,)
//         }
//     }

//     impl Debug for TensorDisplay {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             write!(f, "
//             Tensor {{\n
//                 size: {:#?},\n
//                 device: {:#?},\n
//             }}
//             ",
//             // self.tensor.size(),
//             self.size,
//             // self.tensor.device(),)
//             self.device,)
//         }
//     }

//     impl Deref for TensorDisplay {
//         type Target = Tensor;

//         fn deref(&self) -> &Self::Target {
//             &self.tensor
//         }
//     }

//     impl DerefMut for TensorDisplay {
//         fn deref_mut(&mut self) -> &mut Self::Target {
//             &mut self.tensor
//         }
//     }
// }
// pub struct TensorDisplay {
//     tensor: Tensor,
//     size: Vec<i64>,
//     kind: Kind,
//     max: f64,
//     min: f64,
//     device: Device,
// }

// impl TensorDisplay {
//     pub fn new(ten: Tensor) -> Self {
//         let size = ten.size();
//         let kind = ten.kind();
//         let max = f64::try_from(ten.max()).unwrap();
//         let min = f64::try_from(ten.min()).unwrap();
//         let device = ten.device();
//         Self {
//             tensor: ten,
//             size,
//             kind,
//             max,
//             min,
//             device,
//         }
//     }
// }

// impl Display for TensorDisplay {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "
//         Tensor {{\n
//             size: {:#?},\n
//             device: {:#?},\n
//         }}
//         ",
//         // self.tensor.size(),
//         self.size,
//         // self.tensor.device(),)
//         self.device,)
//     }
// }

// impl Debug for TensorDisplay {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "
//         Tensor {{\n
//             size: {:#?},\n
//             device: {:#?},\n
//         }}
//         ",
//         // self.tensor.size(),
//         self.size,
//         // self.tensor.device(),)
//         self.device,)
//     }
// }

// impl Deref for TensorDisplay {
//     type Target = Tensor;

//     fn deref(&self) -> &Self::Target {
//         &self.tensor
//     }
// }

// impl DerefMut for TensorDisplay {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.tensor
//     }
// }

// impl Display for Tensor {

// }
