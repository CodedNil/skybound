pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod verdeaceae;

pub const TAXON: Taxon = Taxon {
    name: "Bryophytales",
    english_name: "Moss-like",
    description: "These moss-like plants spread as soft mats over bark and other suitable surfaces. \
        Closely packed small growths retain moisture and trap decaying material. \
        They stabilise tiny habitats for insects and other animals living on larger aerial organisms.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
    radius: 0.0,
};
