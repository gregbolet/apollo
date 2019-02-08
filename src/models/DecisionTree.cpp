
#include <map>
#include <string>
#include <sstream>
#include <iostream>

#include "external/nlohmann/json.hpp"
using json = nlohmann::json;

#include "caliper/cali.h"
#include "caliper/common/cali_types.h"

#include "apollo/Apollo.h"
#include "apollo/models/DecisionTree.h"

#define modelName "decisiontree"
#define modelFile __FILE__


int
Apollo::Model::DecisionTree::recursiveTreeWalk(Node *node) {
    // Compare the node->value to the defined comparison values
    // and either dive down a branch or return the choice up.
    if (node->feature->value <= node->value_LEQ) {
        if (node->left_child == nullptr) {
            return node->recommendation;
        } else {
            return recursiveTreeWalk(node->left_child);
        }
    }
    if (node->feature->value > node->value_GRT) {
        if (node->right_child == nullptr) {
            return node->recommendation;
        } else {
            return recursiveTreeWalk(node->right_child);
        }
    }
    return node->recommendation;
} //end: recursiveTreeWalk(...)   [function]


int
Apollo::Model::DecisionTree::getIndex(void)
{
    // Keep choice around for easier debugging, if needed:
    static int choice = -1;

    iter_count++;
    if (configured == false) {
        if (iter_count < 10) {
            fprintf(stderr, "[ERROR] DecisionTree::getIndex() called prior to"
                " model configuration. Defaulting to index 0.\n");
            fflush(stderr);
        } else if (iter_count == 10) {
            fprintf(stderr, "[ERROR] DecisionTree::getIndex() has still not been"
                    " configured. Continuing default behavior without further"
                    " error messages.\n");
        }        
        // Since we're not configured yet, return a default:
        choice = 0;
        return choice;
    }

    // Refresh the values for each feature mentioned in the decision tree.
    // This gives us coherent tree behavior for this iteration, even if the model
    // gets replaced or other values are coming into Caliper during this process,
    // values being evaluated wont change halfway through walking the tree:
    bool converted_ok = true;
    for (Feature *feat : tree_features) {
        feat->value_variant = cali_get(feat->cali_id);
        feat->value         = cali_variant_to_double(feat->value_variant, &converted_ok);
        if (not converted_ok) {
            fprintf(stderr, "== APOLLO: [ERROR] Unable to convert feature to a double!\n");
        }
    }

    // Find the recommendation of the decision tree:
    choice = recursiveTreeWalk(tree_head);

    return choice;
}


void
Apollo::Model::DecisionTree::configure(
        Apollo      *apollo_ptr,
        int          numPolicies,
        std::string  model_definition)
{
    //NOTE: Make sure to grab the lock from the calling code:
    //          std::lock_guard<std::mutex> lock(model->modelMutex);

    apollo       = apollo_ptr;
    policy_count = numPolicies;

    if (configured == true) {
        //This is a RE-configuration. Remove previous configuration.
        configured = false;
        tree_head = nullptr;
        for (Node *node : tree_nodes) {
            if (node != nullptr) { free(node); }
        }
        tree_nodes.clear();
        for (Feature *feat : tree_features) {
            if (feat != nullptr) { free(feat); }
        }
        tree_features.clear();
        model_def = "";
        iter_count = 0;
    }

    // Construct a decisiontree for this model_definition.
    if (model_definition == "") {
        fprintf(stderr, "[WARNING] Cannot successfully configure"
                " with a NULL or empty model definition.\n");
        model_def = "";
        configured = false;
        return;
    }

    model_def = model_definition;

    //TODO: Wrap this loop in a DFS recursive unrolling of the JSON tree:
    
    // #####
    // #
    // #
    // FOR this NODE these will get plucked from the Decision Tree:
    cali_id_t   feat_id;
    std::string feat_name = "";
    double      leq_val = 0.0;
    double      grt_val = 0.0;
    int         recc_val = 0;

    // Scan to see if we have this feature in our accelleration structure::
    bool found = false;
    for (Feature *feat : tree_features) {
        if (feat->name == feat_name) {
            found = true;
            break;
        }
    }
    if (not found) {
        // add it
        feat_id = cali_find_attribute(feat_name.c_str());
        if (feat_id == CALI_INV_ID) {
            fprintf(stderr, "== APOLLO: "
            "[ERROR] DecisionTree contains features no present in Caliper data.\n"
                "\tThis is likely do to an error in the Apollo Controller logic.\n"
                "\tTerminating.\n");
            fflush(stderr);
            exit(EXIT_FAILURE);
        } else {
            Feature *feat = new Feature();
            // NOTE: feat->value_variant and feat->value are filled
            //       before being used for traversal in getIndex()
            feat->cali_id = feat_id;
            feat->name    = feat_name;
            tree_features.push_back(feat);
        }
    }

    configured = true;
    return;
}
//
// ----------
//
// BELOW: Boilerplate code to manage instances of this model:
//


Apollo::Model::DecisionTree::DecisionTree()
{
    iter_count = 0;
}

Apollo::Model::DecisionTree::~DecisionTree()
{
    return;
}

extern "C" Apollo::Model::DecisionTree*
APOLLO_model_create_decisiontree(void)
{
    return new Apollo::Model::DecisionTree();
}


extern "C" void
APOLLO_model_destroy_decisiontree(
        Apollo::Model::DecisionTree *tree_ref)
{
    delete tree_ref;
    return;
}



    /* DEPRECATED: This is the old hand-rolled model encoding format.
     *
    std::istringstream model(model_def);
    typedef std::stringstream  unpack_str;
    std::string        line;

    int num_features = 0;

    // Find out how many features are named:
    std::getline(model, line);
    unpack_str(line) >> num_features;

    // Load in the feature names:   (values are fetched by getIndex())
    for (int i = 0; i < num_features; i++) {
        std::getline(model, line);
        tree_features.push_back(std::string(line));
    }

    // Load in the rules:
    int         line_id   = -1;
    std::string line_feat = NULL;
    std::string line_op   = NULL;
    double      line_val  = 0;

    Node *node = nullptr;

    while (std::getline(model, line)) {
        unpack_str(line) >> line_id >> line_feat >> line_op >> line_val;

        auto tree_find = tree_nodes.find(line_id);
        //
        // Either we are adding to an existing node, or a new one, either way
        // make 'node' point to the correct object instance.
        if (tree_find == tree_nodes.end()) { 
                node = new Node(apollo, line_id, line_feat.c_str());
                tree_nodes.emplace(line_id, node);
        } else {
                node = tree_find->second;
        }

        // Set the components of this node, give the content of this line.
        // NOTE: apollo, node_id, and node_feat have already been set.
        if (line_op == "<=") {
            break;
        } else if (line_op == ">") {
            break;
        } else if (line_op == "=") {
            break;
        } else {
            fprintf(stderr, "ERROR: Unable to process decision"
                    " tree model definition (bad operator"
                    " encountered).\n");
            configured = false;
            return; 
        } // end: if(node_op...)

        //...
    }
    */

