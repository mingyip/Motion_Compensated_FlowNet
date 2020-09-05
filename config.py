import argparse

def configs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        help="Path to data directory.",
                        default='../data/')
    parser.add_argument('--load_path',
                        type=str,
                        help="Path to saved model.",
                        default='data/saver/')
    parser.add_argument('--training_instance',
                        type=str,
                        help="Specific saved model to load. A new one will be generated if empty.",
                        default='')
    parser.add_argument('--batch_size',
                        type=int,
                        help="Training batch size.",
                        default=8)
    parser.add_argument('--logging_interval',
                        type=int,
                        help="Logging frequency",
                        default=10)
    parser.add_argument('--initial_learning_rate',
                        type=float,
                        help="Initial learning rate.",
                        default=5e-4)
    parser.add_argument('--learning_rate_decay',
                        type=float,
                        help='Rate at which the learning rate is decayed.',
                        default=0.9)
    parser.add_argument('--smoothness_weight',
                        type=float,
                        help='Weight for the smoothness term in the loss function.',
                        default=0.001)
    parser.add_argument('--image_height',
                        type=int,
                        help="Image height.",
                        default=256)
    parser.add_argument('--image_width',
                        type=int,
                        help="Image width.",
                        default=256)
    parser.add_argument('--no_batch_norm',
                        action='store_true',
                        help='If true, batch norm will not be performed at each layer',
                        default=False)
    
    # Args for testing only.
    
    parser.add_argument('--test_skip_frames',
                        action='store_true',
                        help='If true, input images will be 4 frames apart.')
    
    parser.add_argument('--test_sequence',
                        type=str,
                        help="Name of the test sequence.",
                        default='outdoor_day2')
    parser.add_argument('--gt_path',
                        type=str,
                        help='Path to optical flow ground truth npz file.',
                        default='../data/outdoor_day2_gt_flow_dist.npz')
    parser.add_argument('--test_plot',
                        action='store_true',
                        help='If true, the flow predictions will be visualized during testing.',
                        default=True)
    parser.add_argument('--save_test_output',
                        action='store_true',
                        help='If true, output flow will be saved to a npz file.',
                        default='')
    
    
    args = parser.parse_args()
    return args