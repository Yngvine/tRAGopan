
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.


#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *



#show: doc => acl(doc,
  anonymous: false,
  title: [NLP Capstone Project: tRAGopan (Topografic Retrieval Augmented Generation of Pamplona and Navarre)],
  authors: make-authors(
    (
      name: "Igor Vons, Endika Aguirre and Maria Ines Haddad",
      affiliation: []
    ),
  ),
)


#abstract[
  #lorem(50)
]


= Introduction

#lorem(80)

#lorem(80)

#lorem(80)


// Uncomment this to include your bibliography:
// #add-bib-resource(read("references.bib"))
// #print-acl-bibliography()
