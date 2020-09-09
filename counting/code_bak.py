            if cx > 0.5*w and cy > 0.7*h:
                if min_dist not in ["movement_5","movement_7","movement_2"]:
                    # if is_same_direction(track_center, base_movements['movement_5'][0], 30):
                    #     min_idx = 'movement_5'
                    # elif is_same_direction(track_center, base_movements['movement_7'][0], 30):
                    #     min_idx = 'movement_7'
                    # elif is_same_direction(track_center, base_movements['movement_2'][0], 30):
                    #     min_idx = 'movement_2'
                   
            if cx < 0.5*w and cy < 0.4*h:
                if min_dist not in ["movement_3","movement_10","movement_1"]:
                    if is_same_direction(track_center, base_movements['movement_3'][0], 30):
                        min_idx = 'movement_3'
                    elif is_same_direction(track_center, base_movements['movement_10'][0], 30):
                        min_idx = 'movement_10'
                    elif is_same_direction(track_center, base_movements['movement_1'][0], 30):
                        min_idx = 'movement_1'
            if ex < 0.3*w and ey < 0.6*h :
                if min_dist not in ["movement_3","movement_5","movement_11"]:
                    if is_same_direction(track_center, base_movements['movement_3'][0], 30):
                        min_idx = 'movement_3'
                    elif is_same_direction(track_center, base_movements['movement_5'][0], 30):
                        min_idx = 'movement_5'
                    elif is_same_direction(track_center, base_movements['movement_11'][0], 30):
                        min_idx = 'movement_11'
            if ex < 0.5*w and ey > 0.7*h :
                if min_dist not in ["movement_6","movement_1","movement_12"]:
                    if is_same_direction(track_center, base_movements['movement_6'][0], 30):
                        min_idx = 'movement_6'
                    elif is_same_direction(track_center, base_movements['movement_1'][0], 30):
                        min_idx = 'movement_1'
                    elif is_same_direction(track_center, base_movements['movement_12'][0], 30):
                        min_idx = 'movement_12'
            if cx <0.3*w and cy > 0.5*h :
                if min_dist not in ["movement_6","movement_4","movement_8"]:
                    if is_same_direction(track_center, base_movements['movement_6'][0], 30):
                        min_idx = 'movement_6'
                    elif is_same_direction(track_center, base_movements['movement_4'][0], 30):
                        min_idx = 'movement_4'
                    elif is_same_direction(track_center, base_movements['movement_8'][0], 30):
                        min_idx = 'movement_8'
            if ex>0.5*w and ey <0.33*h :
                if min_dist not in ["movement_4","movement_2","movement_9"]:
                    if is_same_direction(track_center, base_movements['movement_4'][0], 30):
                        min_idx = 'movement_4'
                    elif is_same_direction(track_center, base_movements['movement_2'][0], 30):
                        min_idx = 'movement_2'
                    elif is_same_direction(track_center, base_movements['movement_9'][0], 30):
                        min_idx = 'movement_9'