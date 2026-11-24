// Synchronous multi-robot Gazebo adapter. One batch advances one shared world.
#include <gz/sim/TestFixture.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/msgs/twist.pb.h>
#include <gz/msgs/marker.pb.h>
#include <gz/transport/Node.hh>
#include <nlohmann/json.hpp>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using namespace gz;
using json = nlohmann::json;

struct RobotState {
  sim::Model model;
  sim::Link body;
  std::vector<sim::Entity> collisions;
  std::set<sim::Entity> wheel_collisions;
  transport::Node::Publisher publisher;
  size_t received = 0;
};

int main(int argc, char **argv) {
  if (argc != 2) { std::cerr << "usage: mosaic-worker WORLD.sdf\n"; return 2; }
  try {
    const char *env = std::getenv("MOSAIC_ROBOTS");
    if (!env) throw std::runtime_error("MOSAIC_ROBOTS is required");
    const auto descriptors = json::parse(env);
    const auto groundArray = json::parse(std::getenv("MOSAIC_GROUND_MODELS") ? std::getenv("MOSAIC_GROUND_MODELS") : "[\"ground\"]");
    const std::set<std::string> grounds = groundArray.get<std::set<std::string>>();
    if (!descriptors.is_object() || descriptors.empty()) throw std::runtime_error("Expected robot descriptors");
    transport::Node node;
    std::map<std::string, RobotState> robots;
    std::map<sim::Entity, std::string> owners;
    std::mutex mutex;
    std::condition_variable delivered;
    for (auto it = descriptors.begin(); it != descriptors.end(); ++it) {
      const std::string name = it.key();
      auto &r = robots[name];
      const auto topic = "/model/" + name + "/cmd_vel";
      r.publisher = node.Advertise<msgs::Twist>(topic);
      node.Subscribe<msgs::Twist>(topic, [&, name](const msgs::Twist &) {
        std::lock_guard<std::mutex> lock(mutex);
        ++robots.at(name).received; delivered.notify_all();
      });
    }
    auto velocity = [&](const std::string &name, double v, double w) {
      auto &r = robots.at(name);
      msgs::Twist command; command.mutable_linear()->set_x(v); command.mutable_angular()->set_z(w);
      std::unique_lock<std::mutex> lock(mutex);
      const auto before = r.received;
      if (!r.publisher.Publish(command)) throw std::runtime_error("Velocity publication failed");
      if (!delivered.wait_for(lock, std::chrono::seconds(2), [&]{return r.received > before;}))
        throw std::runtime_error("Velocity delivery timeout");
    };
    sim::TestFixture fixture(argv[1]);
    bool initialized = false;
    json state;
    fixture.OnPreUpdate([&](const sim::UpdateInfo &, sim::EntityComponentManager &ecm) {
      if (initialized) return;
      for (auto &[name, r] : robots) {
        r.model = sim::Model(ecm.EntityByComponents(sim::components::Model(), sim::components::Name(name)));
        if (!r.model.Valid(ecm)) throw std::runtime_error("Robot not found: " + name);
        r.body = sim::Link(r.model.LinkByName(ecm, descriptors.at(name).at("body").get<std::string>()));
        if (!r.body.Valid(ecm)) throw std::runtime_error("Body not found: " + name);
        r.body.EnableVelocityChecks(ecm);
        for (auto link : r.model.Links(ecm))
          for (auto collision : sim::Link(link).Collisions(ecm)) {
            r.collisions.push_back(collision);
            auto linkName = ecm.Component<sim::components::Name>(link)->Data();
            std::transform(linkName.begin(), linkName.end(), linkName.begin(),
                [](unsigned char c) { return std::tolower(c); });
            if (linkName.find("wheel") != std::string::npos) r.wheel_collisions.insert(collision);
            ecm.CreateComponent(collision, sim::components::ContactSensorData());
          }
      }
      ecm.Each<sim::components::Model, sim::components::Name, sim::components::ParentEntity>(
        [&](const sim::Entity &id, const sim::components::Model *, const sim::components::Name *name,
            const sim::components::ParentEntity *parent) {
          if (!ecm.Component<sim::components::World>(parent->Data())) return true;
          for (auto link : sim::Model(id).Links(ecm))
            for (auto c : sim::Link(link).Collisions(ecm)) owners[c] = name->Data();
          return true;
        });
      initialized = true;
    });
    fixture.OnPostUpdate([&](const sim::UpdateInfo &info, const sim::EntityComponentManager &ecm) {
      json states = json::object();
      for (const auto &[name, r] : robots) {
        auto pose = sim::worldPose(r.body.Entity(), ecm);
        auto v = r.body.WorldLinearVelocity(ecm).value_or(math::Vector3d::Zero);
        auto w = r.body.WorldAngularVelocity(ecm).value_or(math::Vector3d::Zero);
        std::set<std::string> contacts;
        std::set<std::string> body_contacts;
        for (auto id : r.collisions) {
          auto data = ecm.Component<sim::components::ContactSensorData>(id);
          if (!data) throw std::runtime_error("Contact feedback unavailable");
          for (const auto &contact : data->Data().contact()) {
            auto a = owners.find(contact.collision1().id());
            auto b = owners.find(contact.collision2().id());
            auto other = (a != owners.end() && a->second == name) ? b : a;
            if (other != owners.end() && other->second != name) {
              if (!grounds.count(other->second)) contacts.insert(other->second);
              if (!r.wheel_collisions.count(id)) body_contacts.insert(other->second);
            }
          }
        }
        states[name] = {{"position", {pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()}},
          {"rpy", {pose.Rot().Roll(), pose.Rot().Pitch(), pose.Rot().Yaw()}},
          {"linear_velocity", {v.X(), v.Y(), v.Z()}}, {"angular_velocity", {w.X(), w.Y(), w.Z()}},
          {"contacts", contacts}, {"body_contacts", body_contacts}};
      }
      json objects = json::object();
      ecm.Each<sim::components::Model, sim::components::Name, sim::components::ParentEntity>(
        [&](const sim::Entity &id, const sim::components::Model *, const sim::components::Name *name,
            const sim::components::ParentEntity *parent) {
          if (!ecm.Component<sim::components::World>(parent->Data()) || robots.count(name->Data())) return true;
          auto p = sim::worldPose(id, ecm);
          objects[name->Data()] = {p.Pos().X(),p.Pos().Y(),p.Pos().Z(),p.Rot().Roll(),p.Rot().Pitch(),p.Rot().Yaw()};
          return true;
        });
      state = {{"robots", states}, {"objects", objects}, {"iterations", info.iterations},
               {"sim_time", std::chrono::duration<double>(info.simTime).count()}};
    });
    fixture.Finalize();
    auto server = fixture.Server();
    server->SetUpdatePeriod(std::chrono::nanoseconds(0));
    const auto settle = std::getenv("MOSAIC_SETTLE_STEPS") ? std::stoul(std::getenv("MOSAIC_SETTLE_STEPS")) : 800;
    if (settle < 1 || settle > 10000) throw std::runtime_error("Invalid settling steps");
    if (!server->Run(true, settle, false)) throw std::runtime_error("Gazebo initialization failed");
    std::cout << "NAV_JSON " << json({{"ready", true}, {"state", state}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      try {
        const auto request = json::parse(line);
        const auto op = request.at("op").get<std::string>();
        if (op == "close") break;
        if (op != "step") throw std::runtime_error("Only batch step and close are supported");
        const auto &actions = request.at("velocities");
        if (!actions.is_object() || actions.size() != robots.size()) throw std::runtime_error("Every robot needs an action");
        const auto count = request.at("steps").get<int>();
        if (count < 1 || count > 10000) throw std::runtime_error("Invalid iteration count");
        // Validate the complete batch before publishing any command.
        for (const auto &[name, r] : robots) {
          const auto a = actions.at(name).get<std::vector<double>>();
          if (a.size() != 2 || !std::isfinite(a[0]) || !std::isfinite(a[1])) throw std::runtime_error("Invalid velocity");
          const double vmax = descriptors.at(name).at("max_linear");
          const double wmax = descriptors.at(name).at("max_angular");
          if (std::abs(a[0]) > vmax + 1e-6 || std::abs(a[1]) > wmax + 1e-6) throw std::runtime_error("Velocity exceeds robot limits");
        }
        for (const auto &[name, r] : robots) velocity(name, actions.at(name).at(0), actions.at(name).at(1));
        // Marker service is one-way and GUI-only. Its absence never blocks the
        // simulation clock or changes collision geometry.
        for (const auto &item : request.value("markers", json::array())) {
          msgs::Marker marker;
          marker.set_ns("mosaic"); marker.set_id(item.at("id"));
          marker.set_action(item.value("delete", false) ? msgs::Marker::DELETE_MARKER : msgs::Marker::ADD_MODIFY);
          marker.set_type(item.value("type", "sphere") == "line" ? msgs::Marker::LINE_STRIP : msgs::Marker::SPHERE);
          marker.mutable_pose()->mutable_orientation()->set_w(1.0);
          const auto p = item.value("position", std::vector<double>{0,0,0});
          marker.mutable_pose()->mutable_position()->set_x(p.at(0));
          marker.mutable_pose()->mutable_position()->set_y(p.at(1));
          marker.mutable_pose()->mutable_position()->set_z(p.at(2));
          const double scale = item.value("scale", 0.35);
          marker.mutable_scale()->set_x(scale); marker.mutable_scale()->set_y(scale); marker.mutable_scale()->set_z(scale);
          const auto color = item.at("color").get<std::vector<double>>();
          auto *material = marker.mutable_material(); material->set_lighting(false);
          for (auto *c : {material->mutable_diffuse(), material->mutable_ambient(), material->mutable_emissive()}) {
            c->set_r(color.at(0)); c->set_g(color.at(1)); c->set_b(color.at(2)); c->set_a(color.at(3));
          }
          for (const auto &point : item.value("points", json::array())) {
            auto *v = marker.add_point(); v->set_x(point.at(0)); v->set_y(point.at(1)); v->set_z(point.at(2));
          }
          node.Request("/marker", marker);
        }
        if (!server->Run(true, count, false)) throw std::runtime_error("Gazebo stepping failed");
        std::cout << "NAV_JSON " << state.dump() << std::endl;
      } catch (const std::exception &e) {
        for (const auto &[name, r] : robots) velocity(name, 0, 0);
        std::cout << "NAV_JSON " << json({{"error", e.what()}}).dump() << std::endl;
      }
    }
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
