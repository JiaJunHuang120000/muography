while true; do
	tmux kill-session -t janus_ptrg
	cp /home/arratialab/Documents/DroneScripts/ptrg.txt /home/arratialab/Documents/Janus_5202_4.2.4_20251007_linux/bin/Janus_Config.txt
	sleep 5
	python3 test_janus_recording_ptrg.py 60
	sleep 5
	tmux kill-session -t janus_ptrg
	sleep 5

	tmux kill-session -t janus_tlogic
	cp /home/arratialab/Documents/DroneScripts/cosmic.txt /home/arratialab/Documents/Janus_5202_4.2.4_20251007_linux/bin/Janus_Config.txt
	sleep 5
	python3 test_janus_recording_tlogic.py 120
	sleep 5
	tmux kill-session -t janus_tlogic
	sleep 5
done