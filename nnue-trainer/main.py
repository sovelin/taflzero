from torch import tensor, float32
from torch.utils.data import DataLoader
from src.model.chess_dataset import ChessDataset
from src.model.plain_chess_dataset import PlainChessDataset
from src.model.train_data_manager import TrainDataManager
from src.networks.halfkp.data_manager import HalfKPDataManager
from src.networks.halfkp.network import HalfKPNetwork
from src.networks.hnefatafl.data_manager import HnefataflNetworkDataManager
from src.networks.hnefatafl.network import HnefataflNetwork
from src.networks.opposite.data_manager import OppositeNetworkDataManager
from src.networks.opposite.network import OppositeNetwork
from src.networks.simple.data_manager import SimpleNetworkDataManager, SCALE
from src.networks.simple.network import SimpleNetwork, SimpleDeepNetwork
from src.train import train, calculate_wdl_eval_loss, calculate_wdl_eval_loss_dataset


def get_positions_distribution(count: int):
    return round(count * 0.9), round(count * 0.1)

def evaluate_position_simple(fen):
    nnue = HnefataflNetwork(32)
    manager = HnefataflNetworkDataManager()
    nnue.load_weights(37, "trains/hnefatafl-363x32-15M")
    nnue.eval()
    nnue_input = manager.calculate_nnue_input_layer(fen)
    nnue_input = tensor(nnue_input, dtype=float32)

    nnue.print_acc_like_rust(nnue_input.unsqueeze(0))

    print(nnue(nnue_input).item() * SCALE)


def evaluate_position_opposite(fen):
    nnue = OppositeNetwork(8)
    manager = OppositeNetworkDataManager()
    nnue.load_weights(5, "trains/768x8-opposite-try")
    nnue.eval()
    nnue_input_us, nnue_input_them, turn = manager.calculate_nnue_input_layer(fen)
    nnue_input_us = tensor(nnue_input_us, dtype=float32)
    nnue_input_them = tensor(nnue_input_them, dtype=float32)
    turn = tensor([turn], dtype=float32)
    nnue_input_us = nnue_input_us.unsqueeze(0)
    nnue_input_them = nnue_input_them.unsqueeze(0)
    print(nnue(nnue_input_us, nnue_input_them, turn).item() * SCALE)

def evaluate_position_simple_deep(fen):
    nnue = SimpleDeepNetwork(128, 16)
    manager = SimpleNetworkDataManager()
    nnue.load_weights(2, "trains/768x128x16_130MSelfPlay_full-relu")
    nnue.eval()
    nnue_input = manager.calculate_nnue_input_layer(fen)
    nnue_input = tensor(nnue_input, dtype=float32)
    print(nnue(nnue_input).item() * SCALE)

def evaluate_position_halfkp(fen):
    nnue = HalfKPNetwork(128)
    manager = HalfKPDataManager()
    nnue.load_weights(1, "halfkp")
    nnue.eval()
    nnue_input1, nnue_input2 = manager.calculate_nnue_input_layer(fen)
    nnue_input1 = tensor(nnue_input1, dtype=float32)
    nnue_input2 = tensor(nnue_input2, dtype=float32)
    nnue_input1 = nnue_input1.unsqueeze(0)
    nnue_input2 = nnue_input2.unsqueeze(0)
    print(nnue(nnue_input1, nnue_input2).item())

def create_singlethreaded_data_loader(manager: TrainDataManager, path: str):
    dataset = ChessDataset(path, manager)
    return DataLoader(dataset, batch_size=16384, num_workers=0)

def create_data_loader(manager: TrainDataManager, path: str, positions_count: int, is_plain_dataset: bool = False):
    dataset = ChessDataset(path, manager, positions_count) \
        if not is_plain_dataset \
        else PlainChessDataset(path, manager, positions_count)
    return DataLoader(dataset, batch_size=16384, num_workers=0)


def run_opposite_train_nnue(
    hidden_size: int,
    train_dataset_path: str,
    validation_dataset_path: str,
    train_directory,
    positions_count: int,
    is_plain_dataset: bool = False
):
    # evaluate_position_opposite("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    # evaluate_position_opposite("1nbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/rNBQKBN1 w - - 0 1")
    # evaluate_position_opposite("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    # return None

    train_count, validation_count = get_positions_distribution(positions_count)

    manager = OppositeNetworkDataManager()

    train(
        OppositeNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path, train_count, is_plain_dataset),
        create_data_loader(manager, validation_dataset_path, validation_count, is_plain_dataset),
        train_directory
    )

def run_simple_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory,
        positions_count: int
):
    # evaluate_position_simple("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    # evaluate_position_simple("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # evaluate_position_simple("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1")
    # return None

    train_count, validation_count = get_positions_distribution(positions_count)

    manager = SimpleNetworkDataManager()

    train(
        SimpleNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path, train_count),
        create_data_loader(manager, validation_dataset_path, validation_count),
        train_directory
    )

def run_halfkp_train_nnue(
        hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory
):
    manager = HalfKPDataManager()

    train(
        HalfKPNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path),
        create_data_loader(manager, validation_dataset_path),
        train_directory
    )


def run_simple_deep_train_nnue(
        hidden_size: int,
        second_hidden_size: int,
        train_dataset_path: str,
        validation_dataset_path: str,
        train_directory,
        positions_count: int
):
    # evaluate_position_simple_deep("r1b2rk1/pppp1ppp/2n5/2b3Bq/3p4/2PB1N2/PP3PPP/RN1Q1RK1 w - - 9 11")
    # evaluate_position_simple_deep("2k5/8/8/8/8/8/8/2KBN3 w - - 0 1")
    # evaluate_position_simple_deep("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # evaluate_position_simple_deep("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # evaluate_position_simple_deep("1nb1kbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQha - 0 1")
    # evaluate_position_simple_deep("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    # evaluate_position_simple_deep("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1")
    # evaluate_position_simple_deep("4k3/8/8/8/8/8/QQQQQQQQ/QQQQKQQQ w HAha - 0 1")
    # evaluate_position_simple_deep("2kr1br1/p1p2p2/4p2p/3p1np1/8/1PB2P2/P4P1P/2R3RK b - - 3 22")
    # evaluate_position_simple_deep("2kr1br1/p1p2p2/4p2p/3p1np1/8/1PB2P2/P4P1P/2R3RK w - - 3 22")
    # evaluate_position_simple_deep("r2qkb1r/ppp2pp1/2n1p3/4P1Pp/2BP2bK/3Q4/PBP2P2/2R2R2 b kq - 2 17")
    # return None

    train_count, validation_count = get_positions_distribution(positions_count)

    manager = SimpleNetworkDataManager()

    train(
        SimpleDeepNetwork(hidden_size, second_hidden_size),
        create_data_loader(manager, train_dataset_path, train_count),
        create_data_loader(manager, validation_dataset_path, validation_count),
        train_directory
    )


SHOULD_TRAIN_SIMPLE = False
SHOULD_TRAIN_HALFKP = False
SHOULD_TRAIN_SIMPLE_DEEP = False
TRAINS_DIR = "trains"


def run_hnefatafl_train_nnue(
    hidden_size: int,
    train_dataset_path: str,
    validation_dataset_path: str,
    train_directory,
    positions_count: int
):
    train_count, validation_count = get_positions_distribution(positions_count)

    manager = HnefataflNetworkDataManager()

    train(
        HnefataflNetwork(hidden_size),
        create_data_loader(manager, train_dataset_path, train_count),
        create_data_loader(manager, validation_dataset_path, validation_count),
        train_directory
    )

if __name__ == '__main__':
    # evaluate_position_simple("4aaa4/3a1a5/7a3/a4d4a/a2dddd3a/a2d1kdd1aa/aa2ddd3a/a4d4a/11/5a5/3aaaaa3 d")
    
    run_hnefatafl_train_nnue(
        32,
        "train_data2.csv",
        "validate_data2.csv",
        f"{TRAINS_DIR}/hnefatafl-363x32-15M-gen2",
        8361000
    )
