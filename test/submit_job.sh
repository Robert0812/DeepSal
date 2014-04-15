git pull

tag=`date +%Y-%m-%d.%H:%M`
if [$# -eq 0]
 then 
    msg=$tag
 else 
    msg="$tag_$1"
fi

nohup python test_convnet_msra_3layers.py > "../logs/msra_3layers_$msg.log" &
echo "submitted pid is $!"
