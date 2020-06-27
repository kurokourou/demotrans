for i in $(seq -f "%02g_10" 1 10)
do
echo $i
t2t-decoder \
--data_dir=drive/My\ Drive/vocabs \
--t2t_usr_dir=demotrans/core2core \
--problem=translate_envi_core2core \
--model=transformer \
--decode_hparams="beam_size=5,alpha=1.0"  \
--decode_from_file="temp_input/core_"$i".txt" \
--decode_to_file="temp_output/core_"$i"_trans.txt" \
--hparams_set=transformer_base_single_gpu \
--output_dir=core2core_model
done