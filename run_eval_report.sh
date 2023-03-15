for MODEL in BB BBm BBl DiaL

do

    for SEQ in 0 1 2 3 4 5 6 7
    do
        python /home/chenboc1/localscratch2/chenboc1/Adver_Conv/result/result_figure/scripts/Table_safety_defense.py --sequence_number $SEQ --path_class $MODEL
        parlai eval_model -t dialogue_safety:multiturn -dt test -mf zoo:dialogue_safety/multi_turn/model --split-lines True -bs 40 --no-cuda --rf dec_test/dec_test.text
        mv dec_test/dec_test.text dec_test/dec_test.text.$MODEL.$SEQ
        echo "I will create $MODEL and $SEQ"

    done

done
