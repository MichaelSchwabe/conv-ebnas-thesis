  *		?Zd?c@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?mm?y???!?Dx???D@)f???8???1ԋ??<?B@:Preprocessing2T
Iterator::Root::ParallelMapV2?ՏM?#??!ҍ??T?B@)?ՏM?#??1ҍ??T?B@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeattzލ???!?,鷛%@)?P?????1??b?m$@:Preprocessing2E
Iterator::RootH5???:??!$@m-#@E@)S"?^F??1??E?qN@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip\?3??O??!ܿ??ܿL@)??(@?x?1%?)???@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???ٕv?!??h1?@)???ٕv?1??h1?@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapp??1=a??!ZP??oE@)??	j?f?1??T??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF?̱??N?!?ȁ???)F?̱??N?1?ȁ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.