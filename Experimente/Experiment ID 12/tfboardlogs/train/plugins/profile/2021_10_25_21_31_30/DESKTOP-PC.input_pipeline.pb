  *	?A`??ff@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate4d<J%<??!H!? ??B@)??`?$ͯ?1?????TA@:Preprocessing2T
Iterator::Root::ParallelMapV2?z?V????!???TRLA@)?z?V????1???TRLA@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatEc??l???!TŊ?0@)?I??ǜ?1?u?]/@:Preprocessing2E
Iterator::Root|{נ/???!??^$UC@)??[X7?}?1?и?F@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipO#-??#??!hT??۪N@)?&???Kz?1 P?Q??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapx'?۲?!???D@)?=?N??y?1l??D@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??-Yu?!'e??#D@)??-Yu?1'e??#D@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??DJ?y\?!??????)??DJ?y\?1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.