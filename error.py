28.9 M    Trainable params
609 M     Non-trainable params
637 M     Total params
2,551.988 Total estimated model params size (MB)
710       Modules in train mode
0         Modules in eval mode
[NeMo W 2026-07-08 21:04:02 nemo_logging:364] /usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.

[NeMo W 2026-07-08 21:04:02 nemo_logging:364] /usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/data_connector.py:424: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.

Epoch 0:   0%|                                                                                 | 0/315 [00:00<?, ?it/s][NeMo I 2026-07-08 21:04:02 asr_model:198] CUDA graphs disabled for EncDecRNNTBPEModelWithPrompt::RNNTBPEDecoding::GreedyBatchedRNNTInfer
[NeMo W 2026-07-08 21:04:03 nemo_logging:364] /usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py:537: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))

[NeMo W 2026-07-08 21:04:03 nemo_logging:364] /usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py:537: NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
      warn(NumbaPerformanceWarning(msg))

Traceback (most recent call last):
  File "/workspace/scripts/finetune_nemotron.py", line 72, in <module>
    if __name__=='__main__': os.environ.setdefault('TOKENIZERS_PARALLELISM','false'); main()
                                                                                      ^^^^^^
  File "/workspace/scripts/finetune_nemotron.py", line 70, in main
    model.set_trainer(trainer); print('[train] Starting fine-tuning...'); trainer.fit(model)
                                                                          ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 538, in fit
    call._call_and_handle_interrupt(
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 47, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 574, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 981, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1025, in _run_stage
    self.fit_loop.run()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 205, in run
    self.advance()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/fit_loop.py", line 363, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/training_epoch_loop.py", line 140, in run
    self.advance(data_fetcher)
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/training_epoch_loop.py", line 250, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 183, in run
    closure()
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 144, in __call__
    self._result = self.closure(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 129, in closure
    step_output = self._step_fn()
                  ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/loops/optimization/automatic.py", line 317, in _training_step
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 319, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/lightning/pytorch/strategies/strategy.py", line 390, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/scripts/finetune_nemotron.py", line 52, in new_training_step
    def new_training_step(self,batch,batch_idx): return old_train(add_prompt(batch), batch_idx)
                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/utils/model_utils.py", line 471, in wrap_training_step
    output_dict = wrapped(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 418, in training_step
    loss_value, wer, _, _ = self.joint(
                            ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/common.py", line 1365, in wrapped_call
    outputs = wrapped(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/modules/rnnt.py", line 1577, in forward
    loss_batch = self.loss(
                 ^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/common.py", line 1365, in wrapped_call
    outputs = wrapped(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/losses/rnnt.py", line 488, in forward
    loss = self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py", line 437, in forward
    return self.loss(
           ^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py", line 62, in forward
    loss_func(
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/rnnt.py", line 223, in rnnt_loss_gpu
    status = wrapper.cost_and_grad(
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py", line 252, in cost_and_grad
    return self.compute_cost_and_score(acts, grads, costs, pad_labels, label_lengths, input_lengths)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/parts/numba/rnnt_loss/utils/cuda_utils/gpu_rnnt.py", line 199, in compute_cost_and_score
    gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0](
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 540, in __call__
    return self.dispatcher.call(args, self.griddim, self.blockdim,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 682, in call
    kernel = _dispatcher.Dispatcher._cuda_call(self, *args)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 690, in _compile_for_args
    return self.compile(tuple(argtypes))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 933, in compile
    kernel = _Kernel(self.py_func, argtypes, **self.targetoptions)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 84, in __init__
    cres = compile_cuda(self.py_func, types.void, self.argtypes,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/compiler.py", line 196, in compile_cuda
    cres = compiler.compile_extra(typingctx=typingctx,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 739, in compile_extra
    return pipeline.compile_extra(func)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 439, in compile_extra
    return self._compile_bytecode()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 505, in _compile_bytecode
    return self._compile_core()
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 481, in _compile_core
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 473, in _compile_core
    pm.run(self.state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 363, in run
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 356, in run
    self._runPass(idx, pass_inst, state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 311, in _runPass
    mutated |= check(pss.run_pass, internal_state)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 272, in check
    mangled = func(compiler_state)
              ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typed_passes.py", line 114, in run_pass
    typemap, return_type, calltypes, errs = type_inference_stage(
                                            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typed_passes.py", line 95, in type_inference_stage
    errs = infer.propagate(raise_errors=raise_errors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 1075, in propagate
    errors = self.constraints.propagate(self)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 160, in propagate
    constraint(typeinfer)
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 572, in __call__
    self.resolve(typeinfer, typevars, fnty)
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 595, in resolve
    sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typeinfer.py", line 1569, in resolve_call
    return self.context.resolve_function_type(fnty, pos_args, kw_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/context.py", line 279, in resolve_function_type
    res = self._resolve_user_function_type(func, args, kws)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/context.py", line 335, in _resolve_user_function_type
    return func.get_call_type(self, args, kws)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/types/functions.py", line 314, in get_call_type
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/types/functions.py", line 311, in get_call_type
    sig = temp.apply(nolitargs, nolitkws)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/templates.py", line 358, in apply
    sig = generic(args, kws)
          ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/templates.py", line 621, in generic
    disp, new_args = self._get_impl(args, kws)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/templates.py", line 720, in _get_impl
    impl, args = self._build_impl(cache_key, args, kws)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/typing/templates.py", line 824, in _build_impl
    disp_type.get_call_type(self.context, args, kws)
  File "/usr/local/lib/python3.11/site-packages/numba/core/types/functions.py", line 541, in get_call_type
    self.dispatcher.get_call_template(args, kws)
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 850, in get_call_template
    self.compile_device(tuple(args))
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/dispatcher.py", line 884, in compile_device
    cres = compile_cuda(self.py_func, return_type, args,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/cuda/compiler.py", line 196, in compile_cuda
    cres = compiler.compile_extra(typingctx=typingctx,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 739, in compile_extra
    return pipeline.compile_extra(func)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 439, in compile_extra
    return self._compile_bytecode()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 505, in _compile_bytecode
    return self._compile_core()
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 481, in _compile_core
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler.py", line 473, in _compile_core
    pm.run(self.state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 363, in run
    raise e
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 356, in run
    self._runPass(idx, pass_inst, state)
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_lock.py", line 35, in _acquire_compile_lock
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 311, in _runPass
    mutated |= check(pss.run_pass, internal_state)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/compiler_machinery.py", line 272, in check
    mangled = func(compiler_state)
              ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/numba/core/untyped_passes.py", line 105, in run_pass
    raise TypeError("Signature mismatch: %d argument types given, "
TypeError: Signature mismatch: 2 argument types given, but function takes 1 arguments
Epoch 0:   0%|          | 0/315 [00:04<?, ?it/s]
