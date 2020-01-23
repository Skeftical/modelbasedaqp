for i in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  echo "Executing Python script with sampling ratio $i"
  python verdict_runner_instacart.py -s $i
done
