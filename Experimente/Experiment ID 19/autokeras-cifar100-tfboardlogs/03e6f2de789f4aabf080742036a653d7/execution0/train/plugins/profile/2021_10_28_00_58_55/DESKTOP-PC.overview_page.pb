?&  *	?ʡE?J?@2?
ZIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[0]::BatchV2W??:???!???m??7@)2??8*7??1???u??/@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Mapg??)??!????2@)Z?>?-W??1?J???,@:Preprocessing2?
}Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[0]::BatchV2?u?T??!???k?2@)Z-??DJ??1????r$@:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[1]::BatchV2?f?8???!G??-@)???Co??1?5??#@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[0]::BatchV2::TensorSlice ????c???!??AL?+!@)????c???1??AL?+!@:Preprocessing2?
gIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[0]::BatchV2::TensorSlice Kvl?u??!S??ej @)Kvl?u??1S??ej @:Preprocessing2?
gIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[1]::BatchV2::TensorSlice J}Yک???!?"6???@)J}Yک???1?"6???@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2??S?ƙ?!??a7P@)??S?ƙ?1??a7P@:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV27p??G??!???Yq@)7p??G??1???Yq@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2ٴR???!ꉹ??I@)ٴR???1ꉹ??I@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[1]::BatchV2::TensorSlice ????je??!??F?@)????je??1??F?@:Preprocessing2?
}Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[1]::BatchV2???A{???!?A6???@)$??ŋ???1o???A
@:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Zip???RxФ?!LQ&??0@)?#)?ah??1?	??L
 @:Preprocessing2?
qIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip??????!t'߽;@)?<0???1w??#???:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip
L?u???!	vK?D@){??v? }?1O0?R???:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2?Ry=?t?!?c??????)?Ry=?t?1?c??????:Preprocessing2F
Iterator::Model$a?N"§?!K?-?!@)?p?;j?1?D?l???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismOq??!?V?A?? @)Xr??d?1???c?G??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake(??G???!eH{?FD@)$Di?]?1$.?>d??:Preprocessing2?
lIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake?0???C??!?3?_;@)Z???аX?1??(????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q?o?ƹW@"?
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
Refer to the TF2 Profiler FAQb?94.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.EDESKTOP-PC: Failed to load libcupti (is it installed and accessible?)