?&  *	?p=
?)~@2?
}Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[0]::BatchV2?~2Ƈٿ?!?p?/h?9@)kH?c?C??1
???//@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map??Dg?E??!???=?3@)$??P??1?~$?F?.@:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[0]::BatchV2?׹i3N??!mkX?6@) ??Udt??1?D"??*@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[0]::BatchV2::TensorSlice ?l??<+??!??/?_$@)?l??<+??1??/?_$@:Preprocessing2?
gIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[0]::BatchV2::TensorSlice ?ӝ'????!??k??!@)?ӝ'????1??k??!@:Preprocessing2?
`Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2?q:??!?"01jk@)?q:??1?"01jk@:Preprocessing2?
gIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[1]::BatchV2::TensorSlice b???LL??!G\??u?@)b???LL??1G\??u?@:Preprocessing2?
ZIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zip[1]::BatchV2O;?5Y???!?TsI?R"@)=?K?e???1?MA???@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2/?
Ҍ??!Mn$Fq@)/?
Ҍ??1Mn$Fq@:Preprocessing2?
QIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2N?t"???!;-fi?@)N?t"???1;-fi?@:Preprocessing2?
}Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[1]::BatchV2?M)??Н?!9m???!@)NB?!???1??24@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip[1]::BatchV2::TensorSlice `YiR
???!Y$h??@)`YiR
???1Y$h??@:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Zip |(ђǣ?!???7k @)υ?^????1$.?_@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2L8??+x?!??M<???)L8??+x?1??M<???:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake::Zipˢ?????!	E6@@) ??*Q?v?1??Ͽݕ??:Preprocessing2?
qIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake::Zip????G??!*??j@@)e??7it?1e??wE??:Preprocessing2F
Iterator::ModelC??fڦ?!????D"@)??-</k?1)-?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismg??)??!$?:A? !@)u???f?1?S{?????:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[0]::ParallelMapV2::FiniteTake̴?++M??!+mb??n@@)c?D(ba?1??r??#??:Preprocessing2?
lIterator::Model::MaxIntraOpParallelism::Zip[1]::ParallelMapV2::Map::ParallelMapV2::ParallelMapV2::FiniteTake0???tx??!??7啑@@)??p?QeX?1?G?!????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q$D`??;W@"?
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
Refer to the TF2 Profiler FAQb?92.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.EDESKTOP-PC: Failed to load libcupti (is it installed and accessible?)