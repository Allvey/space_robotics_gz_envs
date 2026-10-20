// Synchronous adapter for the project's original Gazebo world and DiffDrive plugin.
#include <gz/sim/TestFixture.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/World.hh>
#include <gz/sim/components/Static.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/msgs/twist.pb.h>
#include <gz/transport/Node.hh>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>
using namespace gz;
using json = nlohmann::json;

int main(int argc, char **argv) {
  if (argc != 2) { std::cerr << "usage: navigation-worker WORLD.sdf\n"; return 2; }
  try {
    const auto robotName = std::string(std::getenv("NAV_ROBOT"));
    const auto bodyName = std::string(std::getenv("NAV_BODY"));
    transport::Node node;
    auto publisher = node.Advertise<msgs::Twist>("/cmd_vel");
    std::mutex mutex;
    std::condition_variable delivered;
    size_t received = 0;
    node.Subscribe<msgs::Twist>("/cmd_vel", [&](const msgs::Twist &) {
      std::lock_guard<std::mutex> lock(mutex); ++received; delivered.notify_all();
    });
    auto velocity = [&](double v, double w) {
      if (!std::isfinite(v) || !std::isfinite(w)) throw std::runtime_error("Non-finite velocity");
      msgs::Twist cmd; cmd.mutable_linear()->set_x(v); cmd.mutable_angular()->set_z(w);
      std::unique_lock<std::mutex> lock(mutex);
      const auto before = received;
      if (!publisher.Publish(cmd)) throw std::runtime_error("/cmd_vel publication failed");
      if (!delivered.wait_for(lock, std::chrono::seconds(2), [&]{return received > before;}))
        throw std::runtime_error("/cmd_vel transport timeout");
    };
    sim::TestFixture fixture(argv[1]);
    sim::Model robot, marker;
    sim::Link body;
    std::vector<sim::Link> robotLinks, objectLinks;
    std::vector<sim::Joint> joints;
    std::map<sim::Entity, size_t> collisions;
    std::map<sim::Entity, math::Pose3d> initialObjects;
    bool initialized = false, reset = false, moveGoal = false;
    math::Pose3d resetPose, goalPose;
    json state;
    fixture.OnPreUpdate([&](const sim::UpdateInfo &, sim::EntityComponentManager &ecm) {
      if (!initialized) {
        robot = sim::Model(ecm.EntityByComponents(sim::components::Model(), sim::components::Name(robotName)));
        if (!robot.Valid(ecm)) throw std::runtime_error("Original robot not found: " + robotName);
        body = sim::Link(robot.LinkByName(ecm, bodyName));
        if (!body.Valid(ecm)) throw std::runtime_error("Original root link not found: " + bodyName);
        marker = sim::Model(ecm.EntityByComponents(sim::components::Model(), sim::components::Name("navigation_goal")));
        for (auto id : robot.Links(ecm)) {
          sim::Link link(id); link.EnableVelocityChecks(ecm); robotLinks.push_back(link);
          for (auto c : link.Collisions(ecm)) {
            collisions[c] = robotLinks.size()-1;
            ecm.CreateComponent(c, sim::components::ContactSensorData());
          }
        }
        for (auto id : robot.Joints(ecm)) joints.emplace_back(id);
        ecm.Each<sim::components::Model, sim::components::ParentEntity>(
          [&](const sim::Entity &id, const sim::components::Model *, const sim::components::ParentEntity *parent) {
            auto fixed = ecm.Component<sim::components::Static>(id);
            if (id == robot.Entity() || !ecm.Component<sim::components::World>(parent->Data()) || (fixed && fixed->Data())) return true;
            initialObjects[id] = sim::worldPose(id, ecm);
            for (auto link : sim::Model(id).Links(ecm)) objectLinks.emplace_back(link);
            return true;
          });
        initialized = true;
      }
      if (moveGoal) { marker.SetWorldPoseCmd(ecm, goalPose); moveGoal = false; }
      if (reset) {
        robot.SetWorldPoseCmd(ecm, resetPose);
        for (auto &link : robotLinks) { link.SetLinearVelocity(ecm, math::Vector3d::Zero); link.SetAngularVelocity(ecm, math::Vector3d::Zero); }
        for (auto &joint : joints) { joint.ResetPosition(ecm, {0}); joint.ResetVelocity(ecm, {0}); }
        for (const auto &[id, pose] : initialObjects) sim::Model(id).SetWorldPoseCmd(ecm, pose);
        for (auto &link : objectLinks) { link.SetLinearVelocity(ecm, math::Vector3d::Zero); link.SetAngularVelocity(ecm, math::Vector3d::Zero); }
        reset = false;
      }
      // Joint control, limits, inertia, contacts and solver belong to the original plugins.
    });
    fixture.OnPostUpdate([&](const sim::UpdateInfo &info, const sim::EntityComponentManager &ecm) {
      const auto p = sim::worldPose(body.Entity(), ecm);
      const auto v = body.WorldLinearVelocity(ecm).value_or(math::Vector3d::Zero);
      const auto w = body.WorldAngularVelocity(ecm).value_or(math::Vector3d::Zero);
      std::map<std::pair<size_t,std::string>, math::Vector3d> forces;
      auto obstacle = [](const std::string &name) { return name.find("lunar_rock") != std::string::npos || name.find("martian_rock") != std::string::npos; };
      for (const auto &[id, link] : collisions) {
        const auto data = ecm.Component<sim::components::ContactSensorData>(id);
        if (!data) throw std::runtime_error("Gazebo contact feedback unavailable");
        for (const auto &contact : data->Data().contact()) {
          auto other = obstacle(contact.collision1().name()) ? contact.collision1().name() : contact.collision2().name();
          if (!obstacle(other)) continue;
          for (const auto &wrench : contact.wrench()) {
            auto f = wrench.body_1_wrench().force();
            forces[{link,other}] += math::Vector3d(f.x(),f.y(),f.z());
          }
        }
      }
      json contacts = json::array(), objects = json::object();
      for (const auto &[key, f] : forces) contacts.push_back({f.X(),f.Y(),f.Z()});
      if (contacts.empty()) contacts.push_back({0.,0.,0.});
      ecm.Each<sim::components::Model,sim::components::Name,sim::components::ParentEntity>(
        [&](const sim::Entity &id, const sim::components::Model *, const sim::components::Name *name, const sim::components::ParentEntity *parent) {
          if (id == robot.Entity() || id == marker.Entity() || !ecm.Component<sim::components::World>(parent->Data())) return true;
          auto q = sim::worldPose(id,ecm);
          objects[name->Data()] = {q.Pos().X(),q.Pos().Y(),q.Pos().Z(),q.Rot().Roll(),q.Rot().Pitch(),q.Rot().Yaw()};
          return true;
        });
      state = {{"position",{p.Pos().X(),p.Pos().Y(),p.Pos().Z()}}, {"rpy",{p.Rot().Roll(),p.Rot().Pitch(),p.Rot().Yaw()}},
               {"linear_velocity",{v.X(),v.Y(),v.Z()}},{"angular_velocity",{w.X(),w.Y(),w.Z()}},
               {"obstacle_forces",contacts},{"objects",objects},{"iterations",info.iterations},
               {"sim_time",std::chrono::duration<double>(info.simTime).count()}};
      if (marker.Valid(ecm)) {
        const auto q = sim::worldPose(marker.Entity(),ecm);
        state["goal_pose"] = {q.Pos().X(),q.Pos().Y(),q.Pos().Z(),q.Rot().Yaw()};
      }
    });
    fixture.Finalize();
    auto server = fixture.Server();
    if (!server->Run(true,1,false)) throw std::runtime_error("Gazebo initialization failed");
    // Disable wall-clock pacing only. The original SDF physics parameters stay intact.
    server->SetUpdatePeriod(std::chrono::nanoseconds(0));
    std::cout << "NAV_JSON " << json({{"ready",true},{"state",state}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin,line)) {
      try {
        auto request = json::parse(line); auto op = request.at("op").get<std::string>();
        if (op == "close") break;
        if (op == "goal") {
          if (marker.Entity() == sim::kNullEntity) throw std::runtime_error("No goal annotation in this world");
          const auto p = request.at("pose").get<std::vector<double>>();
          goalPose = math::Pose3d(p.at(0),p.at(1),p.at(2),0,0,p.at(3)); moveGoal = true;
        } else {
          if (op == "reset") {
            const auto p = request.at("pose").get<std::vector<double>>();
            resetPose = math::Pose3d(p.at(0),p.at(1),p.at(2),0,0,p.at(3)); reset = true; velocity(0,0);
          } else if (op == "step") {
            const auto a = request.at("velocity").get<std::vector<double>>();
            if (a.size()!=2) throw std::runtime_error("Expected [linear, angular] velocity");
            velocity(a[0],a[1]);
          } else throw std::runtime_error("Unknown operation");
          const auto count = request.at("steps").get<unsigned int>();
          if (count<1 || count>10000) throw std::runtime_error("Invalid iteration count");
          if (!server->Run(true,count,false)) throw std::runtime_error("Gazebo stepping failed");
        }
        std::cout << "NAV_JSON " << state.dump() << std::endl;
      } catch (const std::exception &e) { std::cout << "NAV_JSON " << json({{"error",e.what()}}).dump() << std::endl; }
    }
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
