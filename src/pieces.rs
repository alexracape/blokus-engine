/*
Defines Pieces for Blokus Game
*/
use crate::board::BOARD_SIZE;

pub enum PieceType {
    One,
    Two,
    Right,
    Three,
    Four,
    ShortL,
    Triangle,
    Square,
    ShortStep,
    Five,
    LongL,
    LongStep,
    SquarePlus,
    LongRight,
    Steps,
    Z,
    Hump,
    LongWithSide,
    Plus,
    Crazy,
    T
}

pub const PIECE_TYPES: [PieceType; 21] = [
    PieceType::One, 
    PieceType::Two,
    PieceType::Right,
    PieceType::Three,    
    PieceType::Four,
    PieceType::ShortL,
    PieceType::Triangle,
    PieceType::Square,
    PieceType::ShortStep,
    PieceType::Five,
    PieceType::LongL,
    PieceType::LongStep,
    PieceType::SquarePlus,
    PieceType::LongRight,
    PieceType::Steps,
    PieceType::Z,
    PieceType::Hump,
    PieceType::LongWithSide,
    PieceType::Plus,
    PieceType::Crazy,
    PieceType::T
];

/// A piece variant is a specific orientation of a piece
/// It is a list of bools, where true represents a filled square
/// Offsets is a list of offsets to move a filled square to an anchor
#[derive(Clone, Debug)]
pub struct PieceVariant {
    pub offsets: Vec<usize>,
    pub variant: Vec<bool>,
    pub width: usize,
}

impl PieceVariant {
    pub fn new(shape: Vec<Vec<bool>>) -> PieceVariant {
        let mut offsets = Vec::new();
        let mut variant = Vec::new();
        
        // Build the variant that is fully padded to the right
        for (i, row )in shape.iter().enumerate() {
            for square in row {
                variant.push(*square);
            }

            // Pad rest of the row if not last row
            if i == shape.len() - 1 {
                continue;
            }

            for _ in 0..BOARD_SIZE - row.len() {
                variant.push(false);
            }
        }

        // Store offsets to allign pieces later
        for (i, square) in variant.iter().enumerate() {
            if *square {
                offsets.push(i);
            }
        }
        PieceVariant {
            offsets: offsets,
            variant: variant,
            width: shape[0].len(),
        }
    }
}

impl PartialEq for PieceVariant {
    fn eq(&self, other: &Self) -> bool {
        self.variant == other.variant
    }
}


#[derive(Clone)]
pub struct Piece {
    pub shape: Vec<Vec<bool>>,
    pub points: u32,
    pub variants: Vec<PieceVariant>,
}

impl Piece {

    /// Takes a PieceType and redirects to the correct constructor
    /// Those constructors define the shape and create variant shapes
    pub fn new(piece_type: PieceType) -> Piece {
        let shape = match piece_type {
            PieceType::One => vec![vec![true]],
            PieceType::Two => vec![vec![true, true]],
            PieceType::Right => vec![vec![true, true], vec![false, true]],
            PieceType::Three => vec![vec![true, true, true]],
            PieceType::Four => vec![vec![true, true, true, true]],
            PieceType::ShortL => vec![vec![true, true], vec![true, false], vec![true, false]],
            PieceType::Triangle => vec![vec![true, true, true], vec![false, true, false]],
            PieceType::Square => vec![vec![true, true], vec![true, true]],
            PieceType::ShortStep => vec![vec![true, true, false], vec![false, true, true]],
            PieceType::Five => vec![vec![true, true, true, true, true]],
            PieceType::LongL => vec![vec![true, true, true, true], vec![true, false, false, false]],
            PieceType::LongStep => vec![vec![true, true, true, false], vec![false, false, true, true]],
            PieceType::SquarePlus => vec![vec![true, true, true], vec![true, true, false], vec![true, true, false]],
            PieceType::LongRight => vec![vec![true, true, true], vec![true, false, false], vec![true, false, false]],
            PieceType::Steps => vec![vec![true, true, false], vec![false, true, true], vec![false, false, true]],
            PieceType::Z => vec![vec![true, true, false], vec![false, true, true], vec![false, true, true]],
            PieceType::Hump => vec![vec![true, true, true], vec![true, false, true]],
            PieceType::LongWithSide => vec![vec![true, true, true, true], vec![false, true, false, false]],
            PieceType::Plus => vec![vec![false, true, false], vec![true, true, true], vec![false, true, false]],
            PieceType::Crazy => vec![vec![false, true, false], vec![true, true, true], vec![true, false, false]],
            PieceType::T => vec![vec![true, true, true], vec![false, true, false], vec![false, true, false]]
        };

        Piece {
            shape: shape.clone(),
            points: shape.iter().flatten().filter(|&x| *x).count() as u32,
            variants: Piece::gen_variants(shape.clone()),
        }
    }

     // Rotate a piece 90 degrees
     fn rotate(shape: Vec<Vec<bool>>) -> Vec<Vec<bool>> {
        let mut new_shape = Vec::new();
        for i in 0..shape[0].len() {
            let mut row = Vec::new();
            for j in (0..shape.len()).rev() {
                row.push(shape[j][i]);
            }
            new_shape.push(row);
        }
        
        new_shape
    }

    // Flip a piece over
    fn flip(shape: Vec<Vec<bool>>) -> Vec<Vec<bool>> {
        let mut new_shape = Vec::new();
        for row in shape {
            let mut new_row = Vec::new();
            for square in row.iter().rev() {
                new_row.push(*square);
            }
            new_shape.push(new_row);
        }
        new_shape
    }

    fn gen_variants(shape: Vec<Vec<bool>>) -> Vec<PieceVariant> {
        let mut variants = Vec::new();
        let mut variant_shape = shape.clone();

        // Generate all 8 variants
        for _ in 0..4 {

            let new_variant = PieceVariant::new(variant_shape.clone());
            if !variants.contains(&new_variant) {
                variants.push(new_variant);
            }
            variant_shape = Piece::rotate(variant_shape);
        }
        variant_shape = Piece::flip(shape);
        for _ in 0..4 {

            let new_variant = PieceVariant::new(variant_shape.clone());
            if !variants.contains(&new_variant) {
                variants.push(new_variant);
            }
            variant_shape = Piece::rotate(variant_shape);
        }

        variants
    }
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_creation() {
        let piece = Piece::new(PieceType::One);
        assert_eq!(piece.points, 1);
        assert_eq!(piece.variants, Piece::gen_variants(vec![vec![true]]));

        let piece = Piece::new(PieceType::Two);
        assert_eq!(piece.points, 2);
        assert_eq!(piece.variants, Piece::gen_variants(vec![vec![true, true]]));

        let piece = Piece::new(PieceType::Right);
        assert_eq!(piece.points, 3);
        assert_eq!(piece.variants.len(), 4);

        let piece = Piece::new(PieceType::Crazy);
        assert_eq!(piece.points, 5);
        assert_eq!(piece.variants.len(), 8);
    }

    #[test]
    fn test_variant_creation() {
        let variant = PieceVariant::new(vec![vec![true]]);
        assert_eq!(variant.variant, vec![true]);
        assert_eq!(variant.offsets, vec![0]);
        assert_eq!(variant.width, 1);

        let variant = PieceVariant::new(vec![vec![true], vec![true]]);
        assert_eq!(variant.variant.len(), BOARD_SIZE + 1);
        assert_eq!(variant.offsets, vec![0, BOARD_SIZE]);
        assert_eq!(variant.width, 1);
    }

    #[test]
    fn test_piece_rotation() {
        let shape = vec![vec![true, true]];
        let rotated = Piece::rotate(shape.clone());
        assert_eq!(rotated, vec![vec![true], vec![true]]);

        let shape = vec![vec![true, true], vec![true, false]];
        let rotated = Piece::rotate(shape.clone());
        assert_eq!(rotated, vec![vec![true, true], vec![false, true]]);
    }

    #[test]
    fn test_piece_flip() {
        let shape = vec![vec![true, true]];
        let flipped = Piece::flip(shape.clone());
        assert_eq!(flipped, vec![vec![true, true]]);

        let shape = vec![vec![true, true], vec![true, false]];
        let flipped = Piece::flip(shape.clone());
        assert_eq!(flipped, vec![vec![true, true], vec![false, true]]);
    }

    #[test]
    fn test_piece_variants() {
        let shape = vec![vec![true, true]];
        let variants = Piece::gen_variants(shape.clone());
        assert_eq!(variants.len(), 2);

        let shape = vec![vec![true, true], vec![true, false]];
        let variants = Piece::gen_variants(shape.clone());
        assert_eq!(variants.len(), 4);

        let shape = vec![vec![true, true, true], vec![true, false, false]];
        let variants = Piece::gen_variants(shape.clone());
        assert_eq!(variants.len(), 8);
    }
}
