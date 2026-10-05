class Config:
    SEQ_TEST_EMBEDDING_PKL_PATH = "../data/test_embeddings.pkl"
    SEQ_TRAIN_EMBEDDING_PKL_PATH = "../data/train_embeddings.pkl"
    SEQ_VALID_EMBEDDING_PKL_PATH = "../data/valid_embeddings.pkl"

    TRAIN_STRUCT_EMBEDDING_PATH = "../data/embeddings/train_struct_embeddings.pt"

    VALID_STRUCT_EMBEDDING_PATH = "../data/embeddings/valid_struct_embeddings.pt"

    TEST_STRUCT_EMBEDDING_PATH = "../data/embeddings/test_struct_embeddings.pt"

    GO_OBO_PATH = "../data/go.obo"

    # 模型参数
    ESM_DIM = 2560
    TEXT_DIM = 768
    HIDDEN_DIM = 256
    NUM_GCN_LAYERS = 2
    DROPOUT = 0.3
    STRUCT_DIM = 512

    null_input_dim = 14663

    # 训练参数
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-3
    EARLY_STOP_PATIENCE = 10
    DEVICE = "cuda:1"

    # loss
    LOSS_TYPE = "asymmetric"
    ASL_GAMMA_NEG = 4
    ASL_GAMMA_POS = 1
    ASL_CLIP = 0.05

    NUM_WORKERS = 4

    # auxiliary
    USE_AUXILIARY = False
    NUM_AUXILIARY = 100

    # decoder
    DECODER_LAYERS = 2
    DECODER_HEADS = 8
    NUM_QUERIES = 16

    # scheduler
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6

    # contrastive learning
    CL_WEIGHT_MAX = 0.1
    CL_WARMUP_EPOCHS = 10

    # GO DAG
    MAX_DEPTH = 20

    # 默认任务
    TASK = "bp"
    TASK_CONFIG = {
        "bp": {
            "NUM_LABELS": 18140,
            "BATCH_SIZE": 32,
        },

        "cc": {
            "NUM_LABELS": 2759,
            "BATCH_SIZE": 64,
        },

        "mf": {
            "NUM_LABELS": 6557,
            "BATCH_SIZE": 64,
        }
    }

    @classmethod
    def set_task(cls, task):
        task = task.lower()

        if task not in cls.TASK_CONFIG:
            raise ValueError(
                f"未知 task: {task}，请选择 bp / cc / mf"
            )

        cls.TASK = task
        cls.GO_GRAPH_PATH = (
            f"../data/{task}/go_parents_pair.pkl"
        )
        cls.TRAIN_DATA_PATH = (
            f"../data/{task}/train_{task}_multihot.pkl"
        )
        cls.VAL_DATA_PATH = (
            f"../data/{task}/valid_{task}_multihot.pkl"
        )
        cls.TEST_DATA_PATH = (
            f"../data/{task}/test_{task}_multihot.pkl"
        )
        cls.none_embed_pkl_path = (
            "../data/none_finetuned_go_embeddings/"
            f"{task}_go_pretrained_qwen3_from_all.pkl"
        )
        cls.weitiao_embed_pkl_path = (
            "../data/weitiao_finetuned_go_embeddings/"
            f"{task}_go_pretrained_qwen3_from_all.pkl"
        )

        cls.TEST_RESULTS_PKL = (
            f"./results/{task}/test_results.pkl"
        )

        cls.CHECKPOINT_DIR = (
            f"checkpoints/{task}"
        )
        cls.LOG_DIR = (
            f"logs/{task}"
        )
        cfg = cls.TASK_CONFIG[task]

        cls.NUM_LABELS = cfg["NUM_LABELS"]
        cls.BATCH_SIZE = cfg["BATCH_SIZE"]

        print(f"Current task: {cls.TASK}")

        # print("=" * 50)
        # print(f"Current task: {cls.TASK}")
        # print(f"NUM_LABELS: {cls.NUM_LABELS}")
        # print(f"BATCH_SIZE: {cls.BATCH_SIZE}")
        # print(f"TRAIN_DATA_PATH: {cls.TRAIN_DATA_PATH}")
        # print("=" * 50)


# Config.set_task("bp")