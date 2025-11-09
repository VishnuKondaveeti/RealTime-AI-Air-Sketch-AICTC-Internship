import cv2
import numpy as np
import mediapipe as mp

# ===============================================
# 1. SETUP AND INITIALIZATION
# ===============================================

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Start video capture and set resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Drawing Parameters
canvas = np.zeros((720, 1280, 3), dtype=np.uint8) + 255 # White canvas
current_color = (0, 0, 0) # Black (BGR)
brush_thickness = 10
prev_x, prev_y = 0, 0

# Game Parameters (Tic-Tac-Toe)
# 0: Empty, 1: Player X, 2: Player O
game_state = np.zeros((3, 3), dtype=np.int8) 
current_player = 1  # 1 for 'X', 2 for 'O'
game_over = False
status_message = "Player X's turn"
last_placement_time = 0 
COOLDOWN = 2  # Seconds cooldown between moves

# Global Finger Landmark IDs
TIP_IDS = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, 
           mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, 
           mp_hands.HandLandmark.PINKY_TIP]
LOWER_JOINT_IDS = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, 
                   mp_hands.HandLandmark.PINKY_PIP]

# Game Grid coordinates (Top-Left corner for the grid)
GRID_X_START, GRID_Y_START = 300, 200
GRID_SIZE = 500
CELL_SIZE = GRID_SIZE // 3


# ===============================================
# 2. HELPER FUNCTIONS
# ===============================================

def get_fingers_up(landmarks):
    """Checks which fingers are raised."""
    fingers = []
    if landmarks.landmark[TIP_IDS[0]].x < landmarks.landmark[LOWER_JOINT_IDS[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for i in range(1, 5):
        if landmarks.landmark[TIP_IDS[i]].y < landmarks.landmark[LOWER_JOINT_IDS[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def check_win(board):
    """Checks for a win condition (3 in a row, col, or diag)."""
    for player in [1, 2]:
        if any(all(board[r, c] == player for c in range(3)) for r in range(3)): return player # Rows
        if any(all(board[r, c] == player for r in range(3)) for c in range(3)): return player # Columns
        if all(board[i, i] == player for i in range(3)): return player # Diag 1
        if all(board[i, 2 - i] == player for i in range(3)): return player # Diag 2
    if np.all(board != 0): return 3 # Draw (3)
    return 0 # No winner

def draw_game_board(img, board, grid_x, grid_y, cell_s):
    """Draws the grid lines and the X's and O's on the frame."""
    
    # Draw Grid Lines
    grid_end_x, grid_end_y = grid_x + GRID_SIZE, grid_y + GRID_SIZE
    for i in range(1, 3):
        cv2.line(img, (grid_x + i * cell_s, grid_y), (grid_x + i * cell_s, grid_end_y), (100, 100, 100), 4)
        cv2.line(img, (grid_x, grid_y + i * cell_s), (grid_end_x, grid_y + i * cell_s), (100, 100, 100), 4)

    # Draw X's and O's
    for r in range(3):
        for c in range(3):
            center_x = grid_x + c * cell_s + cell_s // 2
            center_y = grid_y + r * cell_s + cell_s // 2
            
            if board[r, c] == 1: # Player X
                size = cell_s // 3
                cv2.line(img, (center_x - size, center_y - size), (center_x + size, center_y + size), (0, 0, 255), 5)
                cv2.line(img, (center_x + size, center_y - size), (center_x - size, center_y + size), (0, 0, 255), 5)
            elif board[r, c] == 2: # Player O
                cv2.circle(img, (center_x, center_y), cell_s // 3, (255, 0, 0), 5)


# ===============================================
# 3. MAIN PROGRAM LOOP
# ===============================================

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Draw UI elements (Color Palette and Game Status)
    cv2.rectangle(img, (20, 0), (280, 100), (50, 50, 50), -1) # Status box
    cv2.putText(img, status_message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.rectangle(img, (300, 0), (500, 100), (255, 0, 0), -1)    # Blue Draw
    cv2.rectangle(img, (500, 0), (700, 100), (0, 255, 0), -1)    # Green Draw
    cv2.rectangle(img, (700, 0), (900, 100), (0, 0, 255), -1)    # Red Draw
    cv2.rectangle(img, (1000, 0), (1200, 100), (0, 0, 0), -1)   # Clear / Reset Game
    cv2.putText(img, "CLEAR / RESET", (1010, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw Tic-Tac-Toe Board
    draw_game_board(img, game_state, GRID_X_START, GRID_Y_START, CELL_SIZE)

    # Hand Detection and Logic
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            tip_of_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, c = img.shape
            tip_x = int(tip_of_index.x * w)
            tip_y = int(tip_of_index.y * h)
            
            fingers = get_fingers_up(hand_landmarks)

            # --- A. Selection/Game Mode (Index and Middle fingers up: [0, 1, 1, 0, 0]) ---
            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                prev_x, prev_y = 0, 0
                cv2.circle(img, (tip_x, tip_y), 15, (255, 255, 255), 2) # White Selector Ring
                
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                
                # 1. UI Selection Logic (Top 100 pixels)
                if tip_y < 100: 
                    if 300 < tip_x < 500: current_color = (255, 0, 0) 
                    elif 500 < tip_x < 700: current_color = (0, 255, 0)
                    elif 700 < tip_x < 900: current_color = (0, 0, 255)
                    elif 1000 < tip_x < 1200: 
                        canvas = np.zeros((720, 1280, 3), dtype=np.uint8) + 255 # Clear Drawing
                        game_state = np.zeros((3, 3), dtype=np.int8) # Reset Game
                        current_player = 1
                        game_over = False
                        status_message = "Player X's turn (Game Reset)"

                # 2. Tic-Tac-Toe Placement Logic (If game is not over)
                elif not game_over:
                    # Check if finger is inside the grid boundary
                    if (GRID_X_START < tip_x < GRID_X_START + GRID_SIZE and
                        GRID_Y_START < tip_y < GRID_Y_START + GRID_SIZE):
                        
                        # Calculate Row and Column (0, 1, or 2)
                        col = (tip_x - GRID_X_START) // CELL_SIZE
                        row = (tip_y - GRID_Y_START) // CELL_SIZE
                        
                        # Cooldown check to prevent immediate multiple moves
                        if current_time - last_placement_time > COOLDOWN:
                            if game_state[row, col] == 0:
                                game_state[row, col] = current_player
                                last_placement_time = current_time # Update cooldown time

                                # Check win condition
                                winner = check_win(game_state)
                                if winner > 0:
                                    game_over = True
                                    if winner == 1: status_message = "Player X Wins!"
                                    elif winner == 2: status_message = "Player O Wins!"
                                    else: status_message = "It's a Draw!"
                                else:
                                    # Switch player
                                    current_player = 2 if current_player == 1 else 1
                                    status_message = f"Player {'X' if current_player == 1 else 'O'}'s turn"
                                

            # --- B. Drawing Mode (Index finger only up: [0, 1, 0, 0, 0]) ---
            elif fingers[1] == 1 and sum(fingers) == 1:
                cv2.circle(img, (tip_x, tip_y), 10, current_color, cv2.FILLED)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = tip_x, tip_y
                else:
                    cv2.line(canvas, (prev_x, prev_y), (tip_x, tip_y), current_color, brush_thickness)
                    prev_x, prev_y = tip_x, tip_y
            
            # --- C. Idle Mode
            else:
                prev_x, prev_y = 0, 0

    # Display Results
    img_combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
    cv2.imshow("RealTime-AI-Air-Sketch", img_combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
