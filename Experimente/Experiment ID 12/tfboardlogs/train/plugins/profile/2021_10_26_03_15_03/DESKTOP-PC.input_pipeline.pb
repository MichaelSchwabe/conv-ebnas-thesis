  *	?S㥛?c@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?OT6????!ԍ?l.?D@)?=@??̮?1p???;C@:Preprocessing2T
Iterator::Root::ParallelMapV2G?,Ҭ?!?&???A@)G?,Ҭ?1?&???A@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatu/3l???!???+?*@)?u?|?H??1?X	U)@:Preprocessing2E
Iterator::Roott_?lW??!???fwhD@)??h?~?1H??/?J@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipR?????!>=???M@)h?N???t?1???	@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceIC??3t?!;v?Ԅ:	@)IC??3t?1;v?Ԅ:	@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapNE*?-??!?\?m5@E@)?Z}uU?V?1T??*?A??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??Hh˹T?!
??"???)??Hh˹T?1
??"???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.