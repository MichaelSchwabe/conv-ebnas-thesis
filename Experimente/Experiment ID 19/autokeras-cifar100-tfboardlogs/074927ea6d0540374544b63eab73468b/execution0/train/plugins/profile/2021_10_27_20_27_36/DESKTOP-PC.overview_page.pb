?&  *???????@)      @=2?
ZIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[0]::BatchV2?G?z??![v?O?A@)?c?]K???1Wy?-?J7@:Preprocessing2?
}Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[0]::BatchV2m???{???!C?Q?n8@)??ݓ????1??X3?#2@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map?	?c??!{}*??-@)????K7??1?])-?(@:Preprocessing2?
gIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[0]::BatchV2::TensorSlice o???T???!?????>(@)o???T???1?????>(@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[0]::BatchV2::TensorSlice ݵ?|г??!?5z ,@)ݵ?|г??1?5z ,@:Preprocessing2?
}Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[1]::BatchV2?c?ZB??!?=ˋ??@)? ?	???1?????@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[1]::BatchV2::TensorSlice B>?٬???!???X?a@)B>?٬???1???X?a@:Preprocessing2?
gIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[1]::BatchV2::TensorSlice }гY????!?o?,
@)}гY????1?o?,
@:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[1]::BatchV2?{??Pk??!?????@)?&1???1?	vҒ	@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2 ?o_Ι?!?*??E	@) ?o_Ι?1?*??E	@:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2M?O???!?tP-B@)M?O???1?tP-B@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2M?O???!?tP-B@)M?O???1?tP-B@:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Zip??ʡE??!7?%??@)?(??0??1ll?????:Preprocessing2?
qIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip?S㥛???!p55l@@)????Mb??1??Y???:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip ?o_???!&???ZE@)F%u?{?1/????y??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2?~j?t?x?!+*????)?~j?t?x?1+*????:Preprocessing2F
Iterator::Model&S????!??W?Z@)?J?4q?1??P3???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??	h"l??!?>G4?@)?J?4q?1??P3???:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake?-?????!?+6(??E@)??_?Le?1?h''???:Preprocessing2?
lIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake;pΈ????!{}*?@@)-C??6Z?1?
??~???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@qa?3??W@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?96.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.EDESKTOP-PC: Failed to load libcupti (is it installed and accessible?)