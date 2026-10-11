pub use super::BIOLOGY;
pub use super::render;
use crate::scene::life::{Rank, Taxon};

pub mod aeromuscaceae;

pub const TAXON: Taxon = Taxon {
    name: "Aeromuscales",
    english_name: "Air Mosses",
    description: "These moss-like plants spread as soft mats over bark and other suitable surfaces. \
        Closely packed small growths retain moisture and trap decaying material. \
        They stabilise tiny habitats for insects and other animals living on larger aerial organisms.",
    rank: Rank::Order,
    parent: Some(&super::TAXON),
};
