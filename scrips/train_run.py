from multiprocessing import freeze_support
from src.engine.train.loop import train

def main():
    train(

        # Chỉnh sửa dataset và folder đầu ra
        dataset_root=r"C:\Dataset\CUB_200_2011", 
        out_dir="outputs/exp_002",
        #-----------------------------------
        
        num_classes=200,
        epochs=10,
        batch_size=32,
        num_workers=0,
        lr=0.01,
        optim_name="sgd",
        sched_name="step",
        step_size=5,
        gamma=0.1,
        loss_name="ce",
        amp=True,
        log_interval=50,
    )

if __name__ == "__main__":
    freeze_support()
    main()
