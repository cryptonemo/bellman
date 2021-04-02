use std::time::Duration;

use crate::bls::Engine;

use crate::gpu::{LockedFFTKernel, LockedMultiexpKernel};
use crate::SynthesisError;

use scheduler_client::{
    register, schedule_one_of, ResourceAlloc, TaskFunc, TaskRequirements, TaskResult,
};

macro_rules! solver {
    ($class:ident, $kern:ident) => {
        pub struct $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine,
        {
            pub accumulator: Vec<R>,
            kernel: Option<$kern<E>>,
            index: usize,
            log_d: usize,
            call: F,
            num_iter: usize,
        }

        impl<E, F, R> $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine,
        {
            pub fn new(log_d: usize, num_iter: usize, call: F) -> Self {
                $class::<E, F, R> {
                    accumulator: vec![],
                    kernel: None,
                    index: 0,
                    log_d,
                    call,
                    num_iter,
                }
            }

            pub fn solve(
                &mut self,
                mut task_req: Option<TaskRequirements>,
            ) -> Result<(), SynthesisError> {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                // get the scheduler client
                let id = rng.gen::<u32>();
                let client = match register(id, id as _) {
                    Ok(c) => c,
                    Err(e) => return Err(e.into()),
                };

                if let Some(ref mut req) = task_req {
                    if self.num_iter == 1 {
                        for resource_req in req.req.iter_mut() {
                            resource_req.preemptible = false;
                        }
                    }
                }

                schedule_one_of(client, self, task_req, Duration::from_secs(90))
                    .map_err(|e| e.into())
            }
        }

        impl<E, F, R> TaskFunc for $class<E, F, R>
        where
            for<'a> F: FnMut(usize, &'a mut Option<$kern<E>>) -> Option<Result<R, SynthesisError>>,
            E: Engine,
        {
            type Output = ();
            type Error = SynthesisError;

            fn init(&mut self, alloc: Option<&ResourceAlloc>) -> Result<Self::Output, Self::Error> {
                self.kernel.replace($kern::<E>::new(self.log_d, alloc));
                Ok(())
            }
            fn end(&mut self, _: Option<&ResourceAlloc>) -> Result<Self::Output, Self::Error> {
                Ok(())
            }
            fn task(&mut self, _alloc: Option<&ResourceAlloc>) -> Result<TaskResult, Self::Error> {
                if let Some(res) = (self.call)(self.index, &mut self.kernel) {
                    match res {
                        Ok(res) => self.accumulator.push(res),
                        Err(e) => return Err(e),
                    }
                    self.index += 1;
                    Ok(TaskResult::Continue)
                } else {
                    Ok(TaskResult::Done)
                }
            }
        }
    };
}

solver!(FftSolver, LockedFFTKernel);
solver!(MultiexpSolver, LockedMultiexpKernel);
