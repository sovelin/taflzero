#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::board::types::Square;
    use crate::board::utils::get_square_from_algebraic;

    fn expect_exactly_neighbors(expected_algebraic: &[&str], actual_squares: &[Square]) {
        assert_eq!(actual_squares.len(), expected_algebraic.len());
        for alg in expected_algebraic {
            let sq = get_square_from_algebraic(alg, 11);
            assert!(
                actual_squares.contains(&sq),
                "Expect neighbor {alg}, but not in the list: {:?}",
                actual_squares
            );
        }
    }

    mod neighbors {
        use super::*;

        #[test]
        fn left_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let left = get_left_neighbor(sq, 11);
            assert_eq!(left, Some(get_square_from_algebraic("c4", 11)));
        }

        #[test]
        fn left_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("a1", 11);
            let left = get_left_neighbor(sq, 11);
            assert_eq!(left, None);
        }

        #[test]
        fn right_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let right = get_right_neighbor(sq, 11);
            assert_eq!(right, Some(get_square_from_algebraic("e4", 11)));
        }

        #[test]
        fn right_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("k8", 11);
            let right = get_right_neighbor(sq, 11);
            assert_eq!(right, None);
        }

        #[test]
        fn top_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let top = get_top_neighbor(sq, 11);
            assert_eq!(top, Some(get_square_from_algebraic("d5", 11)));
        }

        #[test]
        fn top_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("e11", 11);
            let top = get_top_neighbor(sq, 11);
            assert_eq!(top, None);
        }

        #[test]
        fn bottom_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let bottom = get_bottom_neighbor(sq, 11);
            assert_eq!(bottom, Some(get_square_from_algebraic("d3", 11)));
        }

        #[test]
        fn bottom_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("e1", 11);
            let bottom = get_bottom_neighbor(sq, 11);
            assert_eq!(bottom, None);
        }

        #[test]
        fn top_left_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let tl = get_top_left_neighbor(sq, 11);
            assert_eq!(tl, Some(get_square_from_algebraic("c5", 11)));
        }

        #[test]
        fn top_left_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("a8", 11);
            let tl = get_top_left_neighbor(sq, 11);
            assert_eq!(tl, None);
        }

        #[test]
        fn top_right_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let tr = get_top_right_neighbor(sq, 11);
            assert_eq!(tr, Some(get_square_from_algebraic("e5", 11)));
        }

        #[test]
        fn top_right_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("k8", 11);
            let tr = get_top_right_neighbor(sq, 11);
            assert_eq!(tr, None);
        }

        #[test]
        fn bottom_left_neighbor_exists() {
            let sq = get_square_from_algebraic("d4", 11);
            let bl = get_bottom_left_neighbor(sq, 11);
            assert_eq!(bl, Some(get_square_from_algebraic("c3", 11)));
        }

        #[test]
        fn bottom_left_neighbor_does_not_exist() {
            let sq = get_square_from_algebraic("a1", 11);
            let bl = get_top_left_neighbor(sq, 11);
            assert_eq!(bl, None);
        }

        mod mass_neighbor_retrieval {
            use super::*;
            use crate::board::utils::{get_all_neighbors, get_vertical_horizontal_neighbors};

            #[test]
            fn vertical_horizontal_center() {
                let sibs =
                    get_vertical_horizontal_neighbors(get_square_from_algebraic("e6", 11), 11);
                expect_exactly_neighbors(&["e7", "e5", "d6", "f6"], &sibs);
            }

            #[test]
            fn vertical_horizontal_edge() {
                let sibs =
                    get_vertical_horizontal_neighbors(get_square_from_algebraic("a6", 11), 11);
                expect_exactly_neighbors(&["a7", "a5", "b6"], &sibs);
            }

            #[test]
            fn vertical_horizontal_corner() {
                let sibs =
                    get_vertical_horizontal_neighbors(get_square_from_algebraic("a1", 11), 11);
                expect_exactly_neighbors(&["a2", "b1"], &sibs);
            }

            #[test]
            fn all_neighbors_center() {
                let sibs = get_all_neighbors(get_square_from_algebraic("e6", 11), 11);
                expect_exactly_neighbors(&["e7", "e5", "d6", "f6", "d7", "f7", "d5", "f5"], &sibs);
            }

            #[test]
            fn all_neighbors_edge() {
                let sibs = get_all_neighbors(get_square_from_algebraic("a6", 11), 11);
                expect_exactly_neighbors(&["a7", "a5", "b6", "b7", "b5"], &sibs);
            }

            #[test]
            fn all_neighbors_corner() {
                let sibs = get_all_neighbors(get_square_from_algebraic("a1", 11), 11);
                expect_exactly_neighbors(&["a2", "b1", "b2"], &sibs);
            }
        }
    }
}
